package httputil

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// makeStreamResult creates an *ai.StreamResult from a list of StepEvents.
func makeStreamResult(evs ...engine.StepEvent) *ai.StreamResult {
	ch := make(chan engine.StepEvent, len(evs))
	for _, e := range evs {
		ch <- e
	}
	close(ch)
	return ai.NewStreamResult(ch)
}

// TestNewSSEWriter_SetsRequiredHeaders verifies all required SSE headers are set.
func TestNewSSEWriter_SetsRequiredHeaders(t *testing.T) {
	rr := httptest.NewRecorder()
	NewSSEWriter(rr)

	h := rr.Header()
	cases := []struct{ key, want string }{
		{"Content-Type", "text/event-stream"},
		{"Cache-Control", "no-cache"},
		{"Connection", "keep-alive"},
		{"x-vercel-ai-ui-message-stream", "v1"},
	}
	for _, c := range cases {
		if got := h.Get(c.key); got != c.want {
			t.Errorf("header %q: expected %q, got %q", c.key, c.want, got)
		}
	}
}

// TestNewSSEWriter_WritesData verifies data written to sseWriter appears in response.
func TestNewSSEWriter_WritesData(t *testing.T) {
	rr := httptest.NewRecorder()
	sw := NewSSEWriter(rr)

	_, err := sw.Write([]byte("data: hello\n\n"))
	if err != nil {
		t.Fatalf("Write returned error: %v", err)
	}
	if !strings.Contains(rr.Body.String(), "data: hello") {
		t.Errorf("expected body to contain 'data: hello', got: %s", rr.Body.String())
	}
}

// TestStreamToResponse_SetsHeadersAndStreams verifies StreamToResponse sets SSE
// headers and returns full assistant text.
func TestStreamToResponse_SetsHeadersAndStreams(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "Hello "},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "HTTP"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	rr := httptest.NewRecorder()
	text := StreamToResponse(rr, sr, "msg-http")

	// Headers
	if ct := rr.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected Content-Type=text/event-stream, got %q", ct)
	}
	if cc := rr.Header().Get("Cache-Control"); cc != "no-cache" {
		t.Errorf("expected Cache-Control=no-cache, got %q", cc)
	}
	if hdr := rr.Header().Get("x-vercel-ai-ui-message-stream"); hdr != "v1" {
		t.Errorf("expected x-vercel-ai-ui-message-stream=v1, got %q", hdr)
	}

	// Text
	if text != "Hello HTTP" {
		t.Errorf("expected text=%q, got %q", "Hello HTTP", text)
	}

	// Body contains SSE chunks
	body := rr.Body.String()
	if !strings.Contains(body, `"type":"start"`) {
		t.Errorf("expected SSE body to contain start chunk, got: %s", body)
	}
	if !strings.Contains(body, `"type":"finish"`) {
		t.Errorf("expected SSE body to contain finish chunk, got: %s", body)
	}
	if !strings.Contains(body, "[DONE]") {
		t.Errorf("expected SSE body to contain [DONE], got: %s", body)
	}
}

// TestStreamToResponse_FlushBehavior verifies that the sseWriter flushes when
// the ResponseWriter implements http.Flusher.
func TestStreamToResponse_FlushBehavior(t *testing.T) {
	// httptest.ResponseRecorder implements http.Flusher; verify no panic.
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "flush"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	rr := httptest.NewRecorder()
	StreamToResponse(rr, sr, "msg-flush")
	// No panic means flush succeeded.
}

// TestNewSSEWriter_NonFlusherWriter verifies NewSSEWriter works when ResponseWriter
// does not implement http.Flusher.
func TestNewSSEWriter_NonFlusherWriter(t *testing.T) {
	type noFlushWriter struct {
		http.ResponseWriter
	}
	rr := httptest.NewRecorder()
	nfw := &noFlushWriter{ResponseWriter: rr}
	sw := NewSSEWriter(nfw)
	_, err := sw.Write([]byte("data: test\n\n"))
	if err != nil {
		t.Errorf("Write on non-flusher writer returned error: %v", err)
	}
}
