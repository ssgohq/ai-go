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

// WriteMessageMetadata emits a message-metadata chunk with arbitrary metadata.
func (wr *Writer) WriteMessageMetadata(metadata any) {
	WriteSSE(wr.w, Chunk{Type: ChunkMessageMetadata, Fields: map[string]any{"messageMetadata": metadata}})
}

// WriteStartWithMetadata emits a start chunk with message ID and optional metadata.
func (wr *Writer) WriteStartWithMetadata(msgID string, metadata any) {
	fields := map[string]any{"messageId": msgID}
	if metadata != nil {
		fields["messageMetadata"] = metadata
	}
	WriteSSE(wr.w, Chunk{Type: ChunkStart, Fields: fields})
}

// WriteFinishWithReason emits a finish chunk with a finish reason and optional metadata.
// WriteSSE automatically appends the [DONE] terminator for ChunkFinish chunks.
func (wr *Writer) WriteFinishWithReason(finishReason string, metadata any) {
	fields := map[string]any{}
	if finishReason != "" {
		fields["finishReason"] = finishReason
	}
	if metadata != nil {
		fields["messageMetadata"] = metadata
	}
	WriteSSE(wr.w, Chunk{Type: ChunkFinish, Fields: fields})
}

// WriteTransientData emits a custom data-* chunk marked as transient (not persisted).
func (wr *Writer) WriteTransientData(name string, payload any) {
	WriteSSE(wr.w, Chunk{Type: "data-" + name, Fields: map[string]any{
		"data":      payload,
		"transient": true,
	}})
}

// WriteDataWithID emits a custom data-* chunk with an explicit ID for reconciliation.
func (wr *Writer) WriteDataWithID(name, id string, payload any) {
	WriteSSE(wr.w, Chunk{Type: "data-" + name, Fields: map[string]any{
		"id":   id,
		"data": payload,
	}})
}

// WriteAbort emits an abort chunk signaling stream cancellation.
func (wr *Writer) WriteAbort(reason string) {
	fields := map[string]any{}
	if reason != "" {
		fields["reason"] = reason
	}
	WriteSSE(wr.w, Chunk{Type: ChunkAbort, Fields: fields})
}

// WriteSourceURL emits a structured source-url chunk.
func (wr *Writer) WriteSourceURL(sourceID, url, title string) {
	WriteSSE(wr.w, Chunk{Type: ChunkSourceURL, Fields: map[string]any{
		"sourceId": sourceID,
		"url":      url,
		"title":    title,
	}})
}

// WriteSourceDocument emits a structured source-document chunk.
func (wr *Writer) WriteSourceDocument(sourceID, mediaType, title, filename string) {
	WriteSSE(wr.w, Chunk{Type: ChunkSourceDocument, Fields: map[string]any{
		"sourceId":  sourceID,
		"mediaType": mediaType,
		"title":     title,
		"filename":  filename,
	}})
}

// WriteFile emits a file chunk for assistant-provided files.
func (wr *Writer) WriteFile(url, mediaType string) {
	WriteSSE(wr.w, Chunk{Type: ChunkFile, Fields: map[string]any{
		"url":       url,
		"mediaType": mediaType,
	}})
}
