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

type chunkStateHandler func(c Chunk, state *StreamingUIMessageState)

var chunkStateHandlers = map[string]chunkStateHandler{
	ChunkStart:               handleChunkStart,
	ChunkStartStep:           handleChunkStartStep,
	ChunkFinishStep:          handleChunkFinishStep,
	ChunkTextStart:           handleChunkTextStart,
	ChunkTextDelta:           handleChunkTextDelta,
	ChunkTextEnd:             handleChunkTextEnd,
	ChunkReasoningStart:      handleChunkReasoningStart,
	ChunkReasoningDelta:      handleChunkReasoningDelta,
	ChunkReasoningEnd:        handleChunkReasoningEnd,
	ChunkToolInputStart:      handleChunkToolInputStart,
	ChunkToolInputDelta:      handleChunkToolInputDelta,
	ChunkToolInputAvailable:  handleChunkToolInputAvailable,
	ChunkToolOutputAvailable: handleChunkToolOutputAvailable,
	ChunkToolInputError:      handleChunkToolInputError,
	ChunkToolOutputError:     handleChunkToolOutputError,
	ChunkToolOutputDenied:    handleChunkToolOutputDenied,
	ChunkSourceURL:           handleChunkSourceURL,
	ChunkSourceDocument:      handleChunkSourceDocument,
	ChunkFile:                handleChunkFile,
	ChunkFinish:              handleChunkFinish,
	ChunkMessageMetadata:     handleChunkMessageMetadata,
	ChunkError:               handleChunkError,
}

// processChunkIntoState mutates state based on the incoming chunk.
func processChunkIntoState(c Chunk, state *StreamingUIMessageState) {
	if handler := chunkStateHandlers[c.Type]; handler != nil {
		handler(c, state)
	}
}

func handleChunkStart(c Chunk, state *StreamingUIMessageState) {
	if id, ok := c.Fields["messageId"].(string); ok && id != "" {
		state.Message.ID = id
	}
	mergeMessageMetadata(state, c.Fields)
}

func handleChunkStartStep(_ Chunk, state *StreamingUIMessageState) {
	state.Message.Parts = append(state.Message.Parts, UIMessagePart{"type": "step-start"})
}

func handleChunkFinishStep(_ Chunk, state *StreamingUIMessageState) {
	state.ActiveTextParts = make(map[string]*UIMessagePart)
	state.ActiveReasoningParts = make(map[string]*UIMessagePart)
}

func handleChunkTextStart(c Chunk, state *StreamingUIMessageState) {
	id := chunkID(c)
	part := UIMessagePart{"type": "text", "text": ""}
	state.ActiveTextParts[id] = &part
	state.Message.Parts = append(state.Message.Parts, part)
}

func handleChunkTextDelta(c Chunk, state *StreamingUIMessageState) {
	id := chunkID(c)
	appendChunkDeltaToPart(state.ActiveTextParts[id], c.Fields["delta"])
}

func handleChunkTextEnd(c Chunk, state *StreamingUIMessageState) {
	delete(state.ActiveTextParts, chunkID(c))
}

func handleChunkReasoningStart(c Chunk, state *StreamingUIMessageState) {
	id := chunkID(c)
	part := UIMessagePart{"type": "reasoning", "text": ""}
	state.ActiveReasoningParts[id] = &part
	state.Message.Parts = append(state.Message.Parts, part)
}

func handleChunkReasoningDelta(c Chunk, state *StreamingUIMessageState) {
	id := chunkID(c)
	appendChunkDeltaToPart(state.ActiveReasoningParts[id], c.Fields["delta"])
}

func handleChunkReasoningEnd(c Chunk, state *StreamingUIMessageState) {
	delete(state.ActiveReasoningParts, chunkID(c))
}

func handleChunkToolInputStart(c Chunk, state *StreamingUIMessageState) {
	tcID := stringField(c.Fields, "toolCallId")
	if tcID == "" {
		return
	}
	toolName := stringField(c.Fields, "toolName")
	state.PartialToolCalls[tcID] = &partialToolCall{
		ToolName: toolName,
		Index:    len(state.Message.Parts),
	}
	state.Message.Parts = append(state.Message.Parts, UIMessagePart{
		"type":       "tool-invocation",
		"toolCallId": tcID,
		"toolName":   toolName,
		"state":      "input-streaming",
	})
}

func handleChunkToolInputDelta(c Chunk, state *StreamingUIMessageState) {
	tcID := stringField(c.Fields, "toolCallId")
	if tcID == "" {
		return
	}
	ptc := state.PartialToolCalls[tcID]
	if ptc == nil {
		return
	}
	if delta, ok := c.Fields["inputTextDelta"].(string); ok {
		ptc.Text += delta
	}
	if idx := ptc.Index; idx < len(state.Message.Parts) && json.Valid([]byte(ptc.Text)) {
		var parsed any
		if err := json.Unmarshal([]byte(ptc.Text), &parsed); err == nil {
			state.Message.Parts[idx]["input"] = parsed
		}
	}
}

func handleChunkToolInputAvailable(c Chunk, state *StreamingUIMessageState) {
	tcID := stringField(c.Fields, "toolCallId")
	if tcID == "" {
		return
	}
	toolName := stringField(c.Fields, "toolName")
	updateOrAddToolPart(state, tcID, toolName, UIMessagePart{
		"type":       "tool-invocation",
		"toolCallId": tcID,
		"toolName":   toolName,
		"state":      "input-available",
		"input":      c.Fields["input"],
	})
}

func handleChunkToolInputError(c Chunk, state *StreamingUIMessageState) {
	tcID := stringField(c.Fields, "toolCallId")
	toolName := stringField(c.Fields, "toolName")
	updateOrAddToolPart(state, tcID, toolName, UIMessagePart{
		"type":       "tool-invocation",
		"toolCallId": tcID,
		"toolName":   toolName,
		"state":      "output-error",
		"input":      c.Fields["input"],
		"errorText":  c.Fields["errorText"],
	})
}

func handleChunkToolOutputAvailable(c Chunk, state *StreamingUIMessageState) {
	tcID := stringField(c.Fields, "toolCallId")
	if tcID == "" {
		return
	}
	updateToolPartFields(state, tcID, map[string]any{
		"state":  "output-available",
		"output": c.Fields["output"],
	})
}

func handleChunkToolOutputError(c Chunk, state *StreamingUIMessageState) {
	tcID := stringField(c.Fields, "toolCallId")
	if tcID == "" {
		return
	}
	updateToolPartFields(state, tcID, map[string]any{
		"state":     "output-error",
		"errorText": c.Fields["errorText"],
	})
}

func handleChunkToolOutputDenied(c Chunk, state *StreamingUIMessageState) {
	tcID := stringField(c.Fields, "toolCallId")
	if tcID == "" {
		return
	}
	updateToolPartFields(state, tcID, map[string]any{
		"state": "output-denied",
	})
}

func handleChunkSourceURL(c Chunk, state *StreamingUIMessageState) {
	state.Message.Parts = append(state.Message.Parts, UIMessagePart{
		"type":     "source-url",
		"sourceId": c.Fields["sourceId"],
		"url":      c.Fields["url"],
		"title":    c.Fields["title"],
	})
}

func handleChunkSourceDocument(c Chunk, state *StreamingUIMessageState) {
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
}

func handleChunkFile(c Chunk, state *StreamingUIMessageState) {
	part := UIMessagePart{"type": "file"}
	for _, key := range []string{"url", "mediaType", "name"} {
		if v, ok := c.Fields[key].(string); ok && v != "" {
			part[key] = v
		}
	}
	state.Message.Parts = append(state.Message.Parts, part)
}

func handleChunkFinish(c Chunk, state *StreamingUIMessageState) {
	if fr, ok := c.Fields["finishReason"].(string); ok {
		state.FinishReason = fr
	}
	mergeMessageMetadata(state, c.Fields)
}

func handleChunkMessageMetadata(c Chunk, state *StreamingUIMessageState) {
	mergeMessageMetadata(state, c.Fields)
}

func handleChunkError(_ Chunk, _ *StreamingUIMessageState) {
	// Errors are passed through but don't mutate message state.
}

func appendChunkDeltaToPart(part *UIMessagePart, rawDelta any) {
	if part == nil {
		return
	}
	delta, ok := rawDelta.(string)
	if !ok {
		return
	}
	existing, ok := (*part)["text"].(string)
	if !ok {
		existing = ""
	}
	(*part)["text"] = existing + delta
}

// chunkID extracts the "id" field from a chunk.
func chunkID(c Chunk) string {
	return stringField(c.Fields, "id")
}

func stringField(fields map[string]any, key string) string {
	value, ok := fields[key].(string)
	if !ok {
		return ""
	}
	return value
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
