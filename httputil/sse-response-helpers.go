// Package httputil provides HTTP handler helpers for AI SDK SSE streaming responses.
package httputil

import (
	"io"
	"net/http"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/uistream"
)

// sseWriter wraps an http.ResponseWriter to flush after every write.
type sseWriter struct {
	w http.ResponseWriter
	f http.Flusher
}

// Write writes p to the underlying ResponseWriter and flushes immediately.
func (s *sseWriter) Write(p []byte) (int, error) {
	n, err := s.w.Write(p)
	if err == nil && s.f != nil {
		s.f.Flush()
	}
	return n, err
}

// NewSSEWriter wraps w with SSE headers and returns an io.Writer that flushes
// after every write. The required SSE headers are set before returning.
//
// Headers set:
//   - Content-Type: text/event-stream
//   - Cache-Control: no-cache
//   - Connection: keep-alive
//   - x-vercel-ai-ui-message-stream: v1
func NewSSEWriter(w http.ResponseWriter) io.Writer {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("x-vercel-ai-ui-message-stream", "v1")

	var f http.Flusher
	if flusher, ok := w.(http.Flusher); ok {
		f = flusher
	}
	return &sseWriter{w: w, f: f}
}

// StreamToResponse sets SSE headers on w and pipes sr into a UI message stream.
// It is a convenience wrapper around NewSSEWriter + uistream.StreamToWriter.
// Returns the full assistant text for persistence.
func StreamToResponse(
	w http.ResponseWriter,
	sr *ai.StreamResult,
	msgID string,
	opts ...uistream.UIStreamOption,
) string {
	sw := NewSSEWriter(w)
	return uistream.StreamToWriter(sr, sw, msgID, opts...)
}
