package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

func TestNativeLanguageModel_ModelID(t *testing.T) {
	m := NewNativeLanguageModel("gemini-2.5-flash", Config{APIKey: "test"})
	if got := m.ModelID(); got != "gemini-2.5-flash" {
		t.Errorf("ModelID() = %q, want %q", got, "gemini-2.5-flash")
	}
}

func TestNativeLanguageModel_Stream_TextOnly(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}

data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify auth header.
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("x-goog-api-key = %q, want %q", got, "test-key")
		}
		// Verify URL path.
		if !strings.Contains(r.URL.Path, "gemini-2.5-flash:streamGenerateContent") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.URL.Query().Get("alt") != "sse" {
			t.Errorf("missing alt=sse query param")
		}
		// Verify request body.
		body, _ := io.ReadAll(r.Body)
		var req nativeRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to unmarshal request: %v", err)
		}
		if len(req.Contents) != 1 {
			t.Errorf("expected 1 content, got %d", len(req.Contents))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
	}))
	defer srv.Close()

	m := NewNativeLanguageModel("gemini-2.5-flash", Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})

	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("Hi")},
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}

	var events []ai.StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	// Expect: text, usage, text, usage, finish
	var texts []string
	var finishCount int
	for _, ev := range events {
		switch ev.Type {
		case ai.StreamEventTextDelta:
			texts = append(texts, ev.TextDelta)
		case ai.StreamEventFinish:
			finishCount++
			if ev.FinishReason != ai.FinishReasonStop {
				t.Errorf("finish reason = %q, want %q", ev.FinishReason, ai.FinishReasonStop)
			}
		}
	}

	if got := strings.Join(texts, ""); got != "Hello world" {
		t.Errorf("combined text = %q, want %q", got, "Hello world")
	}
	if finishCount != 1 {
		t.Errorf("finish events = %d, want 1", finishCount)
	}
}

func TestNativeLanguageModel_Stream_WithTools(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"London"}}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify tools are in the request body.
		body, _ := io.ReadAll(r.Body)
		var req nativeRequest
		json.Unmarshal(body, &req)
		if len(req.Tools) == 0 {
			t.Error("expected tools in request")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
	}))
	defer srv.Close()

	m := NewNativeLanguageModel("gemini-2.5-flash", Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})

	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("Weather in London?")},
		Tools: []ai.ToolDefinition{{
			Name:        "get_weather",
			Description: "Get weather for a city",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		}},
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}

	var events []ai.StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	var toolCalls int
	var finishReason ai.FinishReason
	for _, ev := range events {
		switch ev.Type {
		case ai.StreamEventToolCallDelta:
			toolCalls++
			if ev.ToolCallName != "get_weather" {
				t.Errorf("tool name = %q, want %q", ev.ToolCallName, "get_weather")
			}
		case ai.StreamEventFinish:
			finishReason = ev.FinishReason
		}
	}

	if toolCalls != 1 {
		t.Errorf("tool call events = %d, want 1", toolCalls)
	}
	if finishReason != ai.FinishReasonToolCalls {
		t.Errorf("finish reason = %q, want %q", finishReason, ai.FinishReasonToolCalls)
	}
}

func TestNativeLanguageModel_Stream_GoogleSearch(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"parts":[{"text":"Search result"}],"role":"model"},"finishReason":"STOP","index":0,"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com","title":"Example"}}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req nativeRequest
		json.Unmarshal(body, &req)

		// Verify Google Search tool is present.
		found := false
		for _, tool := range req.Tools {
			b, _ := json.Marshal(tool)
			if strings.Contains(string(b), "googleSearch") {
				found = true
				break
			}
		}
		if !found {
			t.Error("expected googleSearch tool in request")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
	}))
	defer srv.Close()

	m := NewNativeLanguageModel("gemini-2.5-flash", Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})

	includeThoughts := true
	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("Search for Go")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{
				EnableGoogleSearch: true,
				ThinkingConfig: &ThinkingConfig{
					IncludeThoughts: &includeThoughts,
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}

	var events []ai.StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	var sourceCount int
	for _, ev := range events {
		if ev.Type == ai.StreamEventSource {
			sourceCount++
			if ev.Source.URL != "https://example.com" {
				t.Errorf("source URL = %q, want %q", ev.Source.URL, "https://example.com")
			}
		}
	}
	if sourceCount != 1 {
		t.Errorf("source events = %d, want 1", sourceCount)
	}
}

func TestNativeLanguageModel_Stream_Warnings(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
	}))
	defer srv.Close()

	m := NewNativeLanguageModel("gemini-2.5-flash", Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})

	topK := 40
	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("test")},
		Settings: ai.CallSettings{TopK: &topK},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}

	var events []ai.StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	for _, ev := range events {
		if ev.Type == ai.StreamEventFinish {
			if len(ev.Warnings) == 0 {
				t.Error("expected warnings for topK + google search, got none")
			}
			return
		}
	}
	t.Error("no finish event found")
}

func TestNativeLanguageModel_Stream_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"bad request"}}`))
	}))
	defer srv.Close()

	m := NewNativeLanguageModel("gemini-2.5-flash", Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})

	_, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("test")},
	})
	if err == nil {
		t.Fatal("expected error for 400 response")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("error should mention status 400: %v", err)
	}
}

func TestNativeLanguageModel_ImplementsInterface(t *testing.T) {
	var _ ai.LanguageModel = (*NativeLanguageModel)(nil)
}
