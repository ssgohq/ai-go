package uistream

import (
	"encoding/json"
	"fmt"
	"io"
)

// Writer writes AI SDK UI message stream chunks directly to an io.Writer.
// It is transport-agnostic and does not depend on engine.StepEvent.
// Use it alongside Adapter.Stream, or standalone for custom chunk sequences.
type Writer struct {
	w io.Writer
}

// NewWriter creates a Writer that emits SSE-encoded chunks to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{w: w}
}

// WriteChunk emits a single named chunk with arbitrary key/value fields.
// The type field is always overwritten to match the chunk name.
func (wr *Writer) WriteChunk(typ string, fields map[string]any) {
	payload := make(map[string]any, len(fields)+1)
	for k, v := range fields {
		payload[k] = v
	}
	payload["type"] = typ
	wr.writeSSE(payload)
}

// WriteStart emits the stream start chunk with a message ID.
func (wr *Writer) WriteStart(msgID string) {
	wr.writeSSE(map[string]any{"type": ChunkStart, "messageId": msgID})
}

// WriteFinish emits the stream finish chunk.
func (wr *Writer) WriteFinish() {
	wr.writeSSE(map[string]any{"type": ChunkFinish})
	fmt.Fprintf(wr.w, "data: [DONE]\n\n")
}

// WriteError emits an error chunk with an error message.
func (wr *Writer) WriteError(msg string) {
	wr.writeSSE(map[string]any{"type": ChunkError, "errorText": msg})
}

// WriteData emits a custom data-* chunk.
// name is appended to "data-" to form the chunk type (e.g. name="plan" → type="data-plan").
// payload is JSON-serialized under the "data" key.
func (wr *Writer) WriteData(name string, payload any) {
	wr.writeSSE(map[string]any{
		"type": "data-" + name,
		"data": payload,
	})
}

// Source is a URL reference emitted as part of a source chunk.
type Source struct {
	ID    string `json:"id,omitempty"`
	URL   string `json:"url"`
	Title string `json:"title,omitempty"`
}

// WriteSource emits a source chunk for a single web reference.
func (wr *Writer) WriteSource(s Source) {
	wr.writeSSE(map[string]any{
		"type":   ChunkSource,
		"id":     s.ID,
		"url":    s.URL,
		"title":  s.Title,
	})
}

// WriteSources emits a source chunk containing multiple references.
func (wr *Writer) WriteSources(sources []Source) {
	wr.writeSSE(map[string]any{
		"type":    ChunkSources,
		"sources": sources,
	})
}

func (wr *Writer) writeSSE(payload map[string]any) {
	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	fmt.Fprintf(wr.w, "data: %s\n\n", b)
}
