package uistream

import (
	"io"
)

// CreateUIStreamOptions configures CreateUIMessageStream.
type CreateUIStreamOptions struct {
	MessageID string
	Metadata  any
	OnFinish  func(result UIStreamFinishResult)
	OnError   func(err error) string // return custom error message, or "" for default
}

// UIStreamFinishResult holds info about the completed stream.
type UIStreamFinishResult struct {
	Text         string
	FinishReason string
}

// UIStreamWriter provides write + merge capabilities within a CreateUIMessageStream execute callback.
type UIStreamWriter struct {
	writer     *Writer
	text       string
	lastFinish string
}

// WriteData emits a custom data-* chunk.
func (sw *UIStreamWriter) WriteData(name string, payload any) {
	sw.writer.WriteData(name, payload)
}

// WriteTransientData emits a transient custom data-* chunk.
func (sw *UIStreamWriter) WriteTransientData(name string, payload any) {
	sw.writer.WriteTransientData(name, payload)
}

// WriteSource emits a source chunk.
func (sw *UIStreamWriter) WriteSource(s Source) {
	sw.writer.WriteSource(s)
}

// Merge pipes chunks from a ToUIMessageStream output into this stream.
// The merge respects lifecycle: it skips the start chunk from the merged stream
// (since the outer stream already emitted start) and captures the finish reason
// without emitting finish (the outer stream manages finish).
func (sw *UIStreamWriter) Merge(chunks <-chan Chunk) {
	for c := range chunks {
		switch c.Type {
		case ChunkStart:
			// Skip — outer stream already emitted start.
		case ChunkFinish:
			if fr, ok := c.Fields["finishReason"].(string); ok {
				sw.lastFinish = fr
			}
			// Emit message-metadata from finish if present.
			if md, ok := c.Fields["messageMetadata"]; ok && md != nil {
				sw.writer.WriteMessageMetadata(md)
			}
			// Don't emit finish — outer manages lifecycle.
		default:
			sw.writer.WriteChunk(c.Type, c.Fields)
			// Track text for finish result.
			if c.Type == ChunkTextDelta {
				if delta, ok := c.Fields["delta"].(string); ok {
					sw.text += delta
				}
			}
		}
	}
}

// MergeStreamResult is a convenience that merges a StreamEventer using ToUIMessageStream.
func (sw *UIStreamWriter) MergeStreamResult(sr StreamEventer, msgID string, opts ToUIStreamOptions) {
	chunks := ToUIMessageStream(sr, msgID, opts)
	sw.Merge(chunks)
}

// CreateUIMessageStream creates a managed UI message stream.
// It emits start, runs the execute callback, then emits finish + [DONE].
// The execute callback receives a UIStreamWriter for writing custom data and merging model streams.
func CreateUIMessageStream(w io.Writer, opts CreateUIStreamOptions, execute func(sw *UIStreamWriter) error) {
	// Emit start.
	startFields := map[string]any{"messageId": opts.MessageID}
	if opts.Metadata != nil {
		startFields["messageMetadata"] = opts.Metadata
	}
	WriteSSE(w, Chunk{Type: ChunkStart, Fields: startFields})

	// Create stream writer.
	sw := &UIStreamWriter{
		writer: NewWriter(w),
	}

	// Run execute.
	err := execute(sw)

	// Determine finish reason.
	finishReason := "stop"
	if sw.lastFinish != "" {
		finishReason = sw.lastFinish
	}
	if err != nil {
		errMsg := err.Error()
		if opts.OnError != nil {
			if custom := opts.OnError(err); custom != "" {
				errMsg = custom
			}
		}
		sw.writer.WriteError(errMsg)
		finishReason = "error"
	}

	sw.writer.WriteFinishWithReason(finishReason, nil)

	if opts.OnFinish != nil {
		opts.OnFinish(UIStreamFinishResult{
			Text:         sw.text,
			FinishReason: finishReason,
		})
	}
}
