package openai_compatible_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/open-ai-sdk/ai-go/ai"
	compat "github.com/open-ai-sdk/ai-go/provider/openai_compatible"
)

// sseResponse builds a minimal SSE response body for testing.
func sseResponse(chunks ...string) string {
	var sb strings.Builder
	for _, c := range chunks {
		sb.WriteString("data: ")
		sb.WriteString(c)
		sb.WriteString("\n")
	}
	sb.WriteString("data: [DONE]\n")
	return sb.String()
}

func textChunk(text string) string {
	return fmt.Sprintf(`{"choices":[{"delta":{"content":%q},"finish_reason":null}]}`, text)
}

// newTestServer returns a test HTTP server that echoes request body and sends SSE response.
func newTestServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(handler)
}

func TestNewLanguageModel_ModelID(t *testing.T) {
	m := compat.NewLanguageModel("my-model", compat.Config{
		APIKey:  "test-key",
		BaseURL: "https://example.com/v1",
	})
	if m.ModelID() != "my-model" {
		t.Errorf("expected ModelID=my-model, got %q", m.ModelID())
	}
}

func TestNewLanguageModel_DefaultProviderName(t *testing.T) {
	// The model should construct successfully with an empty provider name (uses default).
	m := compat.NewLanguageModel("test", compat.Config{
		APIKey:  "key",
		BaseURL: "https://example.com/v1",
	})
	if m == nil {
		t.Fatal("expected non-nil model")
	}
}

func TestStream_CustomBaseURL_AndAPIKey(t *testing.T) {
	var receivedAuth string
	var receivedBody map[string]any

	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&receivedBody)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, sseResponse(textChunk("hello")))
	})
	defer srv.Close()

	m := compat.NewLanguageModel("custom-model", compat.Config{
		APIKey:  "my-secret",
		BaseURL: srv.URL,
		Timeout: 5 * time.Second,
	})

	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	}
	ch, err := m.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var events []ai.StreamEvent
	for e := range ch {
		events = append(events, e)
	}

	if receivedAuth != "Bearer my-secret" {
		t.Errorf("expected Bearer my-secret, got %q", receivedAuth)
	}
	if receivedBody["model"] != "custom-model" {
		t.Errorf("expected model=custom-model, got %v", receivedBody["model"])
	}

	hasText := false
	for _, e := range events {
		if e.Type == ai.StreamEventTextDelta && e.TextDelta == "hello" {
			hasText = true
		}
	}
	if !hasText {
		t.Error("expected text delta 'hello'")
	}
}

func TestStream_CustomHeaders(t *testing.T) {
	var receivedHeader string

	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		receivedHeader = r.Header.Get("X-Custom-Key")
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, sseResponse(textChunk("ok")))
	})
	defer srv.Close()

	m := compat.NewLanguageModel("test", compat.Config{
		APIKey:  "key",
		BaseURL: srv.URL,
		Headers: map[string]string{"X-Custom-Key": "custom-value"},
		Timeout: 5 * time.Second,
	})

	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for range ch {
	}

	if receivedHeader != "custom-value" {
		t.Errorf("expected X-Custom-Key=custom-value, got %q", receivedHeader)
	}
}

func TestStream_TransformRequest(t *testing.T) {
	var receivedBody map[string]any

	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, sseResponse(textChunk("ok")))
	})
	defer srv.Close()

	m := compat.NewLanguageModel("test", compat.Config{
		APIKey:  "key",
		BaseURL: srv.URL,
		Timeout: 5 * time.Second,
		TransformRequest: func(req map[string]any) map[string]any {
			req["extra_field"] = "injected"
			return req
		},
	})

	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for range ch {
	}

	if receivedBody["extra_field"] != "injected" {
		t.Errorf("expected extra_field=injected in request body, got %v", receivedBody["extra_field"])
	}
}

func TestStream_CapabilityFlags_StreamUsage(t *testing.T) {
	var receivedBody map[string]any

	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, sseResponse(textChunk("ok")))
	})
	defer srv.Close()

	m := compat.NewLanguageModel("test", compat.Config{
		APIKey:              "key",
		BaseURL:             srv.URL,
		Timeout:             5 * time.Second,
		SupportsStreamUsage: true,
	})

	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for range ch {
	}

	opts, ok := receivedBody["stream_options"].(map[string]any)
	if !ok {
		t.Fatal("expected stream_options in request body when SupportsStreamUsage=true")
	}
	if opts["include_usage"] != true {
		t.Errorf("expected include_usage=true, got %v", opts["include_usage"])
	}
}

func TestStream_NoStreamOptions_WhenCapabilityOff(t *testing.T) {
	var receivedBody map[string]any

	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, sseResponse(textChunk("ok")))
	})
	defer srv.Close()

	m := compat.NewLanguageModel("test", compat.Config{
		APIKey:              "key",
		BaseURL:             srv.URL,
		Timeout:             5 * time.Second,
		SupportsStreamUsage: false,
	})

	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for range ch {
	}

	if receivedBody["stream_options"] != nil {
		t.Errorf("expected no stream_options when SupportsStreamUsage=false, got %v", receivedBody["stream_options"])
	}
}

func TestStream_ProviderName_InError(t *testing.T) {
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":"unauthorized"}`)
	})
	defer srv.Close()

	m := compat.NewLanguageModel("test", compat.Config{
		APIKey:       "bad-key",
		BaseURL:      srv.URL,
		ProviderName: "myProvider",
		Timeout:      5 * time.Second,
	})

	_, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	})
	if err == nil {
		t.Fatal("expected error for 401 response")
	}
	if !strings.Contains(err.Error(), "myProvider") {
		t.Errorf("expected error to mention provider name 'myProvider', got: %v", err)
	}
}
