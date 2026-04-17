package uistream

import (
	"encoding/json"
)

// UIMessagePart represents a typed part accumulated from stream chunks.
// It mirrors the Node AI SDK's UIMessagePart union as a JSON-friendly map.
type UIMessagePart = map[string]any

// StreamingUIMessage represents the assistant message being built from stream chunks.
type StreamingUIMessage struct {
	ID       string          `json:"id"`
	Role     string          `json:"role"`
	Parts    []UIMessagePart `json:"parts"`
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// StreamingUIMessageState holds the mutable state used during stream processing.
// It tracks the in-flight text, reasoning, and tool-call parts.
type StreamingUIMessageState struct {
	Message              StreamingUIMessage
	ActiveTextParts      map[string]*UIMessagePart // keyed by chunk id
	ActiveReasoningParts map[string]*UIMessagePart
	PartialToolCalls     map[string]*partialToolCall
	FinishReason         string
}

type partialToolCall struct {
	Text     string
	Index    int
	ToolName string
}

// NewStreamingUIMessageState creates a new state, optionally continuing from a
// previous assistant message.
func NewStreamingUIMessageState(messageID string, lastMessage *StreamingUIMessage) *StreamingUIMessageState {
	var msg StreamingUIMessage
	if lastMessage != nil && lastMessage.Role == "assistant" {
		// Clone to avoid mutating the caller's message.
		msg = *lastMessage
		parts := make([]UIMessagePart, len(lastMessage.Parts))
		copy(parts, lastMessage.Parts)
		msg.Parts = parts
	} else {
		msg = StreamingUIMessage{
			ID:   messageID,
			Role: "assistant",
		}
	}

	return &StreamingUIMessageState{
		Message:              msg,
		ActiveTextParts:      make(map[string]*UIMessagePart),
		ActiveReasoningParts: make(map[string]*UIMessagePart),
		PartialToolCalls:     make(map[string]*partialToolCall),
	}
}

// ProcessUIMessageStream consumes a channel of Chunks, updates state on each chunk
// to reconstruct the assistant message, and re-emits every chunk on the returned channel.
//
// This is the Go equivalent of AI SDK Node's processUIMessageStream.
func ProcessUIMessageStream(chunks <-chan Chunk, state *StreamingUIMessageState) <-chan Chunk {
	out := make(chan Chunk, 64)
	go func() {
		defer close(out)
		for c := range chunks {
			processChunkIntoState(c, state)
			out <- c
		}
	}()
	return out
}

// processChunkIntoState mutates state based on the incoming chunk.
func processChunkIntoState(c Chunk, state *StreamingUIMessageState) {
	switch c.Type {
	case ChunkStart:
		if id, ok := c.Fields["messageId"].(string); ok && id != "" {
			state.Message.ID = id
		}
		mergeMessageMetadata(state, c.Fields)

	case ChunkStartStep:
		state.Message.Parts = append(state.Message.Parts, UIMessagePart{"type": "step-start"})

	case ChunkFinishStep:
		// Reset active text and reasoning — new step starts fresh.
		state.ActiveTextParts = make(map[string]*UIMessagePart)
		state.ActiveReasoningParts = make(map[string]*UIMessagePart)

	case ChunkTextStart:
		id := chunkID(c)
		part := UIMessagePart{"type": "text", "text": ""}
		state.ActiveTextParts[id] = &part
		state.Message.Parts = append(state.Message.Parts, part)

	case ChunkTextDelta:
		id := chunkID(c)
		if tp := state.ActiveTextParts[id]; tp != nil {
			if delta, ok := c.Fields["delta"].(string); ok {
				(*tp)["text"] = (*tp)["text"].(string) + delta
			}
		}

	case ChunkTextEnd:
		id := chunkID(c)
		delete(state.ActiveTextParts, id)

	case ChunkReasoningStart:
		id := chunkID(c)
		part := UIMessagePart{"type": "reasoning", "text": ""}
		state.ActiveReasoningParts[id] = &part
		state.Message.Parts = append(state.Message.Parts, part)

	case ChunkReasoningDelta:
		id := chunkID(c)
		if rp := state.ActiveReasoningParts[id]; rp != nil {
			if delta, ok := c.Fields["delta"].(string); ok {
				(*rp)["text"] = (*rp)["text"].(string) + delta
			}
		}

	case ChunkReasoningEnd:
		id := chunkID(c)
		delete(state.ActiveReasoningParts, id)

	case ChunkToolInputStart:
		tcID, _ := c.Fields["toolCallId"].(string)
		toolName, _ := c.Fields["toolName"].(string)
		state.PartialToolCalls[tcID] = &partialToolCall{
			ToolName: toolName,
			Index:    len(state.Message.Parts),
		}
		part := UIMessagePart{
			"type":       "tool-invocation",
			"toolCallId": tcID,
			"toolName":   toolName,
			"state":      "input-streaming",
		}
		state.Message.Parts = append(state.Message.Parts, part)

	case ChunkToolInputDelta:
		tcID, _ := c.Fields["toolCallId"].(string)
		if ptc := state.PartialToolCalls[tcID]; ptc != nil {
			if delta, ok := c.Fields["inputTextDelta"].(string); ok {
				ptc.Text += delta
			}
			// Update the part's input with partial JSON if valid.
			if idx := ptc.Index; idx < len(state.Message.Parts) {
				var parsed any
				if json.Valid([]byte(ptc.Text)) {
					_ = json.Unmarshal([]byte(ptc.Text), &parsed)
					state.Message.Parts[idx]["input"] = parsed
				}
			}
		}

	case ChunkToolInputAvailable:
		tcID, _ := c.Fields["toolCallId"].(string)
		toolName, _ := c.Fields["toolName"].(string)
		updateOrAddToolPart(state, tcID, toolName, UIMessagePart{
			"type":       "tool-invocation",
			"toolCallId": tcID,
			"toolName":   toolName,
			"state":      "input-available",
			"input":      c.Fields["input"],
		})

	case ChunkToolOutputAvailable:
		tcID, _ := c.Fields["toolCallId"].(string)
		updateToolPartFields(state, tcID, map[string]any{
			"state":  "output-available",
			"output": c.Fields["output"],
		})

	case ChunkToolInputError:
		tcID, _ := c.Fields["toolCallId"].(string)
		toolName, _ := c.Fields["toolName"].(string)
		updateOrAddToolPart(state, tcID, toolName, UIMessagePart{
			"type":       "tool-invocation",
			"toolCallId": tcID,
			"toolName":   toolName,
			"state":      "output-error",
			"input":      c.Fields["input"],
			"errorText":  c.Fields["errorText"],
		})

	case ChunkToolOutputError:
		tcID, _ := c.Fields["toolCallId"].(string)
		updateToolPartFields(state, tcID, map[string]any{
			"state":     "output-error",
			"errorText": c.Fields["errorText"],
		})

	case ChunkToolOutputDenied:
		tcID, _ := c.Fields["toolCallId"].(string)
		updateToolPartFields(state, tcID, map[string]any{
			"state": "output-denied",
		})

	case ChunkSourceURL:
		state.Message.Parts = append(state.Message.Parts, UIMessagePart{
			"type":     "source-url",
			"sourceId": c.Fields["sourceId"],
			"url":      c.Fields["url"],
			"title":    c.Fields["title"],
		})

	case ChunkSourceDocument:
		part := UIMessagePart{
			"type":      "source-document",
			"sourceId":  c.Fields["sourceId"],
			"mediaType": c.Fields["mediaType"],
			"title":     c.Fields["title"],
		}
		if fn, ok := c.Fields["filename"].(string); ok && fn != "" {
			part["filename"] = fn
		}
		state.Message.Parts = append(state.Message.Parts, part)

	case ChunkFile:
		part := UIMessagePart{"type": "file"}
		for _, key := range []string{"url", "mediaType", "name"} {
			if v, ok := c.Fields[key].(string); ok && v != "" {
				part[key] = v
			}
		}
		state.Message.Parts = append(state.Message.Parts, part)

	case ChunkFinish:
		if fr, ok := c.Fields["finishReason"].(string); ok {
			state.FinishReason = fr
		}
		mergeMessageMetadata(state, c.Fields)

	case ChunkMessageMetadata:
		mergeMessageMetadata(state, c.Fields)

	case ChunkError:
		// Errors are passed through but don't mutate message state.
	}
}

// chunkID extracts the "id" field from a chunk.
func chunkID(c Chunk) string {
	id, _ := c.Fields["id"].(string)
	return id
}

// mergeMessageMetadata merges messageMetadata from chunk fields into the state.
func mergeMessageMetadata(state *StreamingUIMessageState, fields map[string]any) {
	if md := fields["messageMetadata"]; md != nil {
		if raw, err := json.Marshal(md); err == nil {
			state.Message.Metadata = raw
		}
	}
}

// updateOrAddToolPart finds an existing tool part by toolCallId and updates it,
// or appends a new one.
func updateOrAddToolPart(state *StreamingUIMessageState, tcID, toolName string, part UIMessagePart) {
	for i, p := range state.Message.Parts {
		if p["toolCallId"] == tcID && (p["type"] == "tool-invocation" || p["type"] == "dynamic-tool") {
			for k, v := range part {
				state.Message.Parts[i][k] = v
			}
			return
		}
	}
	state.Message.Parts = append(state.Message.Parts, part)
}

// updateToolPartFields finds a tool part by toolCallId and updates specific fields.
func updateToolPartFields(state *StreamingUIMessageState, tcID string, fields map[string]any) {
	for i, p := range state.Message.Parts {
		if p["toolCallId"] == tcID && (p["type"] == "tool-invocation" || p["type"] == "dynamic-tool") {
			for k, v := range fields {
				state.Message.Parts[i][k] = v
			}
			return
		}
	}
}
