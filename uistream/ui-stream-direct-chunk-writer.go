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

// SourceDocumentOpts carries optional v6 fields for source-document chunks.
type SourceDocumentOpts struct {
	Filename         string
	Data             []byte
	ProviderMetadata map[string]any
}

// WriteSourceDocument emits a structured source-document chunk.
// opts may be nil for backward-compatible callers.
func (wr *Writer) WriteSourceDocument(sourceID, mediaType, title string, opts *SourceDocumentOpts) {
	fields := map[string]any{
		"sourceId":  sourceID,
		"mediaType": mediaType,
		"title":     title,
	}
	if opts != nil {
		if opts.Filename != "" {
			fields["filename"] = opts.Filename
		}
		if opts.Data != nil {
			fields["data"] = opts.Data
		}
		fields = withProviderMetadata(fields, opts.ProviderMetadata)
	}
	WriteSSE(wr.w, Chunk{Type: ChunkSourceDocument, Fields: fields})
}

// FileChunkOpts carries optional v6 fields for file chunks.
type FileChunkOpts struct {
	ID               string
	FileID           string
	Data             []byte
	Name             string
	ProviderMetadata map[string]any
}

// WriteFile emits a file chunk for assistant-provided files.
// opts may be nil for backward-compatible callers.
func (wr *Writer) WriteFile(url, mediaType string, opts *FileChunkOpts) {
	fields := map[string]any{
		"url":       url,
		"mediaType": mediaType,
	}
	if opts != nil {
		if opts.ID != "" {
			fields["id"] = opts.ID
		}
		if opts.FileID != "" {
			fields["fileId"] = opts.FileID
		}
		if opts.Data != nil {
			fields["data"] = opts.Data
		}
		if opts.Name != "" {
			fields["name"] = opts.Name
		}
		fields = withProviderMetadata(fields, opts.ProviderMetadata)
	}
	WriteSSE(wr.w, Chunk{Type: ChunkFile, Fields: fields})
}

// ToolChunkOpts carries optional v6 fields for tool-related chunks.
type ToolChunkOpts struct {
	ProviderExecuted *bool
	Dynamic          *bool
	Title            string
	Preliminary      *bool // only meaningful on tool-output-available
}

// applyToolOpts merges non-nil ToolChunkOpts fields into the given fields map.
func applyToolOpts(fields map[string]any, opts *ToolChunkOpts) map[string]any {
	if opts == nil {
		return fields
	}
	if opts.ProviderExecuted != nil {
		fields["providerExecuted"] = *opts.ProviderExecuted
	}
	if opts.Dynamic != nil {
		fields["dynamic"] = *opts.Dynamic
	}
	if opts.Title != "" {
		fields["title"] = opts.Title
	}
	if opts.Preliminary != nil {
		fields["preliminary"] = *opts.Preliminary
	}
	return fields
}

// WriteToolInputError emits a tool-input-error chunk when tool argument parsing fails.
// opts may be nil.
func (wr *Writer) WriteToolInputError(toolCallID, toolName string, input any, errorText string, opts *ToolChunkOpts) {
	fields := applyToolOpts(map[string]any{
		"toolCallId": toolCallID,
		"toolName":   toolName,
		"input":      input,
		"errorText":  errorText,
	}, opts)
	WriteSSE(wr.w, Chunk{Type: ChunkToolInputError, Fields: fields})
}

// WriteToolOutputError emits a tool-output-error chunk when tool execution fails.
// opts may be nil.
func (wr *Writer) WriteToolOutputError(toolCallID, errorText string, opts *ToolChunkOpts) {
	fields := applyToolOpts(map[string]any{
		"toolCallId": toolCallID,
		"errorText":  errorText,
	}, opts)
	WriteSSE(wr.w, Chunk{Type: ChunkToolOutputError, Fields: fields})
}

// WriteToolOutputDenied emits a tool-output-denied chunk when a tool call is rejected.
// opts may be nil.
func (wr *Writer) WriteToolOutputDenied(toolCallID string, opts *ToolChunkOpts) {
	fields := applyToolOpts(map[string]any{
		"toolCallId": toolCallID,
	}, opts)
	WriteSSE(wr.w, Chunk{Type: ChunkToolOutputDenied, Fields: fields})
}

// WriteToolApprovalRequest emits a tool-approval-request chunk for human-in-the-loop flows.
func (wr *Writer) WriteToolApprovalRequest(approvalID, toolCallID, toolName string, args any) {
	WriteSSE(wr.w, Chunk{Type: ChunkToolApprovalRequest, Fields: map[string]any{
		"approvalId": approvalID,
		"toolCallId": toolCallID,
		"toolName":   toolName,
		"args":       args,
	}})
}

// WriteChunkWithProviderMetadata emits a named chunk with arbitrary fields plus
// optional provider-specific metadata. When providerMeta is nil, the
// providerMetadata key is omitted from the output entirely.
func (wr *Writer) WriteChunkWithProviderMetadata(typ string, fields, providerMeta map[string]any) {
	WriteSSE(wr.w, Chunk{Type: typ, Fields: withProviderMetadata(fields, providerMeta)})
}
