package uistream

import (
	"io"
)

// ExecuteFunc is the callback that writes chunks to the stream.
// The Writer is pre-configured; lifecycle (start/finish) is managed by Execute.
type ExecuteFunc func(w *Writer) error

// StreamOptions configures a UI message stream execution.
type StreamOptions struct {
	// MessageID is the assistant message identifier emitted in the start chunk.
	MessageID string

	// Metadata is optional message-level metadata emitted alongside the start chunk.
	Metadata any

	// OnFinish is called after the stream completes with the finish reason ("stop" or "error").
	OnFinish func(finishReason string)
}

// Execute runs a UI message stream with managed lifecycle.
// It emits a start chunk, delegates writing to fn, then emits finish + [DONE].
// If fn returns an error, an error chunk is emitted instead of finish.
func Execute(w io.Writer, opts StreamOptions, fn ExecuteFunc) {
	// Emit start chunk.
	startFields := map[string]any{"messageId": opts.MessageID}
	if opts.Metadata != nil {
		startFields["messageMetadata"] = opts.Metadata
	}
	WriteSSE(w, Chunk{Type: ChunkStart, Fields: startFields})

	wr := NewWriter(w)

	// Run the caller-provided execute function.
	err := fn(wr)

	// Emit finish or error; WriteFinish also emits [DONE].
	finishReason := "stop"
	if err != nil {
		wr.WriteError(err.Error())
		finishReason = "error"
	} else {
		wr.WriteFinish()
	}

	if opts.OnFinish != nil {
		opts.OnFinish(finishReason)
	}
}
