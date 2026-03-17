package uistream

import (
	"encoding/json"
	"strings"
)

// PersistedMessageBuilder accumulates typed parts from stream chunks
// for persistence to a database.
type PersistedMessageBuilder struct {
	textAccum      strings.Builder
	reasoningAccum strings.Builder
	lastSignature  string

	// pendingTool tracks in-progress tool calls keyed by toolCallId
	pendingTool map[string]*toolInvocationPart

	// ordered list of finalized parts
	parts    []any
	metadata json.RawMessage
}

type toolInvocationPart struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
	State      string `json:"state"`
	Input      any    `json:"input,omitempty"`
	Output     any    `json:"output,omitempty"`
}

// NewPersistedMessageBuilder creates a new builder.
func NewPersistedMessageBuilder() *PersistedMessageBuilder {
	return &PersistedMessageBuilder{
		pendingTool: make(map[string]*toolInvocationPart),
	}
}

// ObserveChunk processes a single stream chunk, accumulating state for persistence.
func (b *PersistedMessageBuilder) ObserveChunk(c Chunk) {
	switch c.Type {
	case ChunkTextStart:
		b.textAccum.Reset()

	case ChunkTextDelta:
		if delta, ok := c.Fields["delta"].(string); ok {
			b.textAccum.WriteString(delta)
		}

	case ChunkTextEnd:
		text := b.textAccum.String()
		if text != "" {
			b.parts = append(b.parts, map[string]any{
				"type": "text",
				"text": text,
			})
			b.textAccum.Reset()
		}

	case ChunkReasoningStart:
		b.reasoningAccum.Reset()
		b.lastSignature = ""

	case ChunkReasoningDelta:
		if delta, ok := c.Fields["delta"].(string); ok {
			b.reasoningAccum.WriteString(delta)
		}
		if sig, ok := c.Fields["signature"].(string); ok && sig != "" {
			b.lastSignature = sig
		}

	case ChunkReasoningEnd:
		if sig, ok := c.Fields["signature"].(string); ok && sig != "" {
			b.lastSignature = sig
		}
		reasoning := b.reasoningAccum.String()
		if reasoning != "" {
			part := map[string]any{
				"type":      "reasoning",
				"reasoning": reasoning,
			}
			if b.lastSignature != "" {
				part["signature"] = b.lastSignature
			}
			b.parts = append(b.parts, part)
			b.reasoningAccum.Reset()
			b.lastSignature = ""
		}

	case ChunkToolInputAvailable:
		tcID, _ := c.Fields["toolCallId"].(string)
		toolName, _ := c.Fields["toolName"].(string)
		input := c.Fields["input"]
		if tcID == "" {
			return
		}
		if _, exists := b.pendingTool[tcID]; !exists {
			b.pendingTool[tcID] = &toolInvocationPart{
				ToolCallID: tcID,
				ToolName:   toolName,
				State:      "input-available",
			}
		}
		b.pendingTool[tcID].Input = input
		b.pendingTool[tcID].ToolName = toolName

	case ChunkToolOutputAvailable:
		tcID, _ := c.Fields["toolCallId"].(string)
		output := c.Fields["output"]
		if tcID == "" {
			return
		}
		tool, exists := b.pendingTool[tcID]
		if !exists {
			tool = &toolInvocationPart{ToolCallID: tcID, State: "output-available"}
			b.pendingTool[tcID] = tool
		}
		tool.Output = output
		tool.State = "output-available"
		b.parts = append(b.parts, map[string]any{
			"type":       "tool-invocation",
			"toolCallId": tool.ToolCallID,
			"toolName":   tool.ToolName,
			"state":      tool.State,
			"input":      tool.Input,
			"output":     tool.Output,
		})
		delete(b.pendingTool, tcID)

	case ChunkSourceURL:
		id, _ := c.Fields["sourceId"].(string)
		url, _ := c.Fields["url"].(string)
		title, _ := c.Fields["title"].(string)
		b.parts = append(b.parts, map[string]any{
			"type":  "source-url",
			"id":    id,
			"url":   url,
			"title": title,
		})

	case ChunkSourceDocument:
		id, _ := c.Fields["sourceId"].(string)
		title, _ := c.Fields["title"].(string)
		mediaType, _ := c.Fields["mediaType"].(string)
		b.parts = append(b.parts, map[string]any{
			"type":      "source-document",
			"id":        id,
			"title":     title,
			"mediaType": mediaType,
		})

	case ChunkFile:
		url, _ := c.Fields["url"].(string)
		mediaType, _ := c.Fields["mediaType"].(string)
		name, _ := c.Fields["name"].(string)
		b.parts = append(b.parts, map[string]any{
			"type":      "file",
			"url":       url,
			"mediaType": mediaType,
			"name":      name,
		})

	case ChunkMessageMetadata:
		if meta := c.Fields["messageMetadata"]; meta != nil {
			if raw, err := json.Marshal(meta); err == nil {
				b.metadata = raw
			}
		}

	default:
		// data-* chunks (non-transient)
		if strings.HasPrefix(c.Type, "data-") && !strings.HasPrefix(c.Type, "transient-data-") {
			transient, _ := c.Fields["transient"].(bool)
			if transient {
				return
			}
			name := strings.TrimPrefix(c.Type, "data-")
			data := c.Fields["data"]
			b.parts = append(b.parts, map[string]any{
				"type":        "data",
				"name":        name,
				"data":        data,
				"isTransient": false,
			})
		}
		// transient-data-* → excluded from parts
	}
}

// Content returns the denormalized text content (all text parts joined).
func (b *PersistedMessageBuilder) Content() string {
	var sb strings.Builder
	for _, p := range b.parts {
		m, ok := p.(map[string]any)
		if !ok {
			continue
		}
		if m["type"] == "text" {
			if t, ok := m["text"].(string); ok {
				sb.WriteString(t)
			}
		}
	}
	return sb.String()
}

// Parts returns the serialized JSON array of typed parts.
func (b *PersistedMessageBuilder) Parts() json.RawMessage {
	if len(b.parts) == 0 {
		return nil
	}
	raw, err := json.Marshal(b.parts)
	if err != nil {
		return nil
	}
	return raw
}

// Metadata returns the message metadata as raw JSON (may be nil).
func (b *PersistedMessageBuilder) Metadata() json.RawMessage {
	return b.metadata
}

// MergeWithPersistence returns a MergeOption that feeds every chunk through
// the builder's ObserveChunk method during MergeStreamResult.
func MergeWithPersistence(b *PersistedMessageBuilder) MergeOption {
	return func(c *mergeConfig) {
		c.persistenceBuilder = b
	}
}
