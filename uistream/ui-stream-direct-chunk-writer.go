package uistream

import (
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
// The type field is always set to typ.
func (wr *Writer) WriteChunk(typ string, fields map[string]any) {
	WriteSSE(wr.w, Chunk{Type: typ, Fields: fields})
}

// WriteStart emits the stream start chunk with a message ID.
func (wr *Writer) WriteStart(msgID string) {
	WriteSSE(wr.w, Chunk{Type: ChunkStart, Fields: map[string]any{"messageId": msgID}})
}

// WriteFinish emits the stream finish chunk followed by the [DONE] terminator.
func (wr *Writer) WriteFinish() {
	// WriteSSE already emits [DONE] for ChunkFinish — but only once. We call it
	// directly here to keep the same output as before.
	WriteSSE(wr.w, Chunk{Type: ChunkFinish, Fields: nil})
}

// WriteError emits an error chunk with an error message.
func (wr *Writer) WriteError(msg string) {
	WriteSSE(wr.w, Chunk{Type: ChunkError, Fields: map[string]any{"errorText": msg}})
}

// WriteData emits a custom data-* chunk.
// name is appended to "data-" to form the chunk type (e.g. name="plan" → type="data-plan").
// payload is JSON-serialized under the "data" key.
func (wr *Writer) WriteData(name string, payload any) {
	WriteSSE(wr.w, Chunk{Type: "data-" + name, Fields: map[string]any{"data": payload}})
}

// Source is a URL reference emitted as part of a source chunk.
type Source struct {
	ID    string `json:"id,omitempty"`
	URL   string `json:"url"`
	Title string `json:"title,omitempty"`
}

// WriteSource emits a source chunk for a single web reference.
func (wr *Writer) WriteSource(s Source) {
	WriteSSE(wr.w, Chunk{Type: ChunkSource, Fields: map[string]any{
		"id":    s.ID,
		"url":   s.URL,
		"title": s.Title,
	}})
}

// WriteSources emits a sources chunk containing multiple references.
func (wr *Writer) WriteSources(sources []Source) {
	WriteSSE(wr.w, Chunk{Type: ChunkSources, Fields: map[string]any{"sources": sources}})
}
