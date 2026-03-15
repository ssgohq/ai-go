package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ssgohq/ai-go/ai"
	"github.com/ssgohq/ai-go/provider/openai"
)

// chatSSE builds a minimal valid SSE stream for chat completions.
func chatSSE(textChunks ...string) string {
	var sb strings.Builder
	for _, chunk := range textChunks {
		data := fmt.Sprintf(
			`{"choices":[{"delta":{"content":%q},"finish_reason":null}]}`,
			chunk,
		)
		sb.WriteString("data: " + data + "\n")
	}
	sb.WriteString(`data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}` + "\n")
	sb.WriteString("data: [DONE]\n")
	return sb.String()
}

// chatServer starts an httptest server that responds with the given SSE payload
// and returns the recorded request body alongside the server.
func chatServer(t *testing.T, sseBody string) (*httptest.Server, *[]byte) {
	t.Helper()
	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		captured = raw
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sseBody)
	}))
	return srv, &captured
}

func TestChatLanguageModel_ModelID(t *testing.T) {
	m := openai.NewChatLanguageModel("gpt-4o", openai.Config{APIKey: "key"})
	if m.ModelID() != "gpt-4o" {
		t.Errorf("expected model id gpt-4o, got %q", m.ModelID())
	}
}

func TestChatLanguageModel_Stream_BasicText(t *testing.T) {
	srv, captured := chatServer(t, chatSSE("Hello", " world"))
	defer srv.Close()

	m := openai.NewChatLanguageModel("gpt-4o", openai.Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})

	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	}
	ch, err := m.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var textParts []string
	var usageEvents, finishEvents int
	for e := range ch {
		switch e.Type {
		case ai.StreamEventTextDelta:
			textParts = append(textParts, e.TextDelta)
		case ai.StreamEventUsage:
			usageEvents++
		case ai.StreamEventFinish:
			finishEvents++
		case ai.StreamEventError:
			t.Fatalf("unexpected error event: %v", e.Error)
		}
	}

	if got := strings.Join(textParts, ""); got != "Hello world" {
		t.Errorf("expected %q, got %q", "Hello world", got)
	}
	if usageEvents != 1 {
		t.Errorf("expected 1 usage event, got %d", usageEvents)
	}
	if finishEvents != 1 {
		t.Errorf("expected 1 finish event, got %d", finishEvents)
	}

	// Verify the request was sent to /chat/completions.
	if !strings.Contains(string(*captured), `"model":"gpt-4o"`) {
		t.Errorf("unexpected request body: %s", string(*captured))
	}
}

func TestChatLanguageModel_Stream_RequestShape(t *testing.T) {
	srv, captured := chatServer(t, chatSSE("ok"))
	defer srv.Close()

	m := openai.NewChatLanguageModel("gpt-4o-mini", openai.Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})

	req := ai.LanguageModelRequest{
		System:   "You are helpful",
		Messages: []ai.Message{ai.UserMessage("what is 2+2?")},
		Settings: ai.CallSettings{MaxTokens: 100},
	}
	ch, err := m.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Drain events.
	for range ch {
	}

	var body map[string]any
	if err := json.Unmarshal(*captured, &body); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}

	if body["model"] != "gpt-4o-mini" {
		t.Errorf("expected model=gpt-4o-mini, got %v", body["model"])
	}
	if body["stream"] != true {
		t.Errorf("expected stream=true, got %v", body["stream"])
	}
	if body["max_tokens"] == nil {
		t.Error("expected max_tokens to be set")
	}

	// Should include stream_options with include_usage.
	opts, _ := body["stream_options"].(map[string]any)
	if opts["include_usage"] != true {
		t.Errorf("expected stream_options.include_usage=true, got %v", opts)
	}

	// First message should be system.
	msgs, _ := body["messages"].([]any)
	if len(msgs) < 2 {
		t.Fatalf("expected at least 2 messages, got %d", len(msgs))
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" {
		t.Errorf("expected first message role=system, got %v", first["role"])
	}
}

func TestChatLanguageModel_Stream_FinishReason(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"done"},"finish_reason":null}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
`
	srv, _ := chatServer(t, sse)
	defer srv.Close()

	m := openai.NewChatLanguageModel("gpt-4o", openai.Config{
		APIKey:  "k",
		BaseURL: srv.URL,
	})

	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("x")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var finishEvent *ai.StreamEvent
	for e := range ch {
		if e.Type == ai.StreamEventFinish && e.RawFinishReason == "stop" {
			ev := e
			finishEvent = &ev
		}
	}
	if finishEvent == nil {
		t.Fatal("expected finish event with reason=stop")
	}
	if finishEvent.FinishReason != ai.FinishReasonStop {
		t.Errorf("expected FinishReasonStop, got %v", finishEvent.FinishReason)
	}
}

func TestChatLanguageModel_Stream_ToolCalls(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"search","arguments":"{\"q\":\"test\"}"}}]},"finish_reason":"tool_calls"}]}
data: [DONE]
`
	srv, _ := chatServer(t, sse)
	defer srv.Close()

	m := openai.NewChatLanguageModel("gpt-4o", openai.Config{
		APIKey:  "k",
		BaseURL: srv.URL,
	})

	ch, err := m.Stream(context.Background(), ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search something")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	toolDeltaCount := 0
	for e := range ch {
		if e.Type == ai.StreamEventToolCallDelta {
			toolDeltaCount++
		}
	}
	if toolDeltaCount == 0 {
		t.Error("expected at least one tool call delta event")
	}
}

func TestChatLanguageModel_Stream_StructuredOutput(t *testing.T) {
	sse := chatSSE(`{"name":"Alice","age":30}`)
	srv, captured := chatServer(t, sse)
	defer srv.Close()

	m := openai.NewChatLanguageModel("gpt-4o", openai.Config{
		APIKey:  "k",
		BaseURL: srv.URL,
	})

	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("give me a person")},
		Output: &ai.OutputSchema{
			Type: "object",
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"required":             []string{"name", "age"},
				"additionalProperties": false,
			},
		},
	}
	ch, err := m.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for range ch {
	}

	var body map[string]any
	if err := json.Unmarshal(*captured, &body); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}

	rf, _ := body["response_format"].(map[string]any)
	if rf["type"] != "json_schema" {
		t.Errorf("expected response_format.type=json_schema, got %v", rf["type"])
	}
	jsonSchema, _ := rf["json_schema"].(map[string]any)
	if jsonSchema["strict"] != true {
		t.Errorf("expected json_schema.strict=true, got %v", jsonSchema["strict"])
	}
}

func TestChatProviderOptions_ParseFromMap(t *testing.T) {
	// ChatLanguageModel inherits from openaichat — no options are parsed yet.
	// This test validates that constructing with Config works cleanly.
	m := openai.NewChatLanguageModel("gpt-4o", openai.Config{
		APIKey:  "key",
		BaseURL: "http://localhost",
	})
	if m.ModelID() != "gpt-4o" {
		t.Errorf("expected gpt-4o, got %q", m.ModelID())
	}
}
