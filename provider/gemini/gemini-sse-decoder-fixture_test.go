package gemini

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

func streamFromString(s string) io.ReadCloser {
	return io.NopCloser(strings.NewReader(s))
}

func collectEvents(body io.ReadCloser) []ai.StreamEvent {
	ch := make(chan ai.StreamEvent, 128)
	decodeSSEStream(context.Background(), body, ch)
	var events []ai.StreamEvent
	for e := range ch {
		events = append(events, e)
	}
	return events
}

func TestDecodeSSE_TextOnly(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}
data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	textDeltas := 0
	usageCount := 0
	finishCount := 0
	for _, e := range events {
		switch e.Type {
		case ai.StreamEventTextDelta:
			textDeltas++
		case ai.StreamEventUsage:
			usageCount++
			if e.Usage.PromptTokens != 10 {
				t.Errorf("expected 10 prompt tokens, got %d", e.Usage.PromptTokens)
			}
		case ai.StreamEventFinish:
			finishCount++
		}
	}
	if textDeltas != 2 {
		t.Errorf("expected 2 text deltas, got %d", textDeltas)
	}
	if usageCount != 1 {
		t.Errorf("expected 1 usage event, got %d", usageCount)
	}
	if finishCount != 1 {
		t.Errorf("expected 1 finish event, got %d", finishCount)
	}
}

func TestDecodeSSE_ReasoningDelta(t *testing.T) {
	thought := true
	_ = thought
	sse := `data: {"choices":[{"delta":{"content":"thinking...","thought":true},"finish_reason":null}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	hasReasoning := false
	for _, e := range events {
		if e.Type == ai.StreamEventReasoningDelta {
			hasReasoning = true
			if e.TextDelta != "thinking..." {
				t.Errorf("unexpected reasoning delta: %q", e.TextDelta)
			}
		}
	}
	if !hasReasoning {
		t.Error("expected a reasoning delta event")
	}
}

func TestDecodeSSE_ToolCallDelta(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"search","arguments":"{\"q\":"}}]},"finish_reason":null}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"","function":{"name":"","arguments":"\"hello\"}"}}]},"finish_reason":null}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	toolDeltas := 0
	for _, e := range events {
		if e.Type == ai.StreamEventToolCallDelta {
			toolDeltas++
		}
	}
	if toolDeltas != 2 {
		t.Errorf("expected 2 tool call deltas, got %d", toolDeltas)
	}
}

func TestDecodeSSE_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	sse := `data: {"choices":[{"delta":{"content":"text"}}]}
data: [DONE]
`
	ch := make(chan ai.StreamEvent, 128)
	decodeSSEStream(ctx, streamFromString(sse), ch)

	var events []ai.StreamEvent
	for e := range ch {
		events = append(events, e)
	}

	hasError := false
	for _, e := range events {
		if e.Type == ai.StreamEventError {
			hasError = true
		}
	}
	if !hasError {
		t.Error("expected an error event on context cancellation")
	}
}

func TestSanitizeToolSchemas(t *testing.T) {
	tools := []map[string]any{
		{
			"type": "function",
			"function": map[string]any{
				"name": "test",
				"parameters": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"$ref":                 "#/defs/Foo",
					"properties": map[string]any{
						"q": map[string]any{
							"type":    "string",
							"default": "hello",
						},
					},
				},
			},
		},
	}

	cleaned := sanitizeToolSchemas(tools)
	fn := cleaned[0]["function"].(map[string]any)
	params := fn["parameters"].(map[string]any)

	if _, ok := params["additionalProperties"]; ok {
		t.Error("additionalProperties should have been removed")
	}
	if _, ok := params["$ref"]; ok {
		t.Error("$ref should have been removed")
	}

	props := params["properties"].(map[string]any)
	q := props["q"].(map[string]any)
	if _, ok := q["default"]; ok {
		t.Error("default should have been removed from nested schema")
	}
	if q["type"] != "string" {
		t.Error("type should be preserved")
	}
}

func TestEncodeRequest_SystemAndMessages(t *testing.T) {
	req := ai.LanguageModelRequest{
		System:   "You are helpful",
		Messages: []ai.Message{ai.UserMessage("hi")},
	}
	cr, err := encodeRequest("gemini-2.5-flash", req, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.Model != "gemini-2.5-flash" {
		t.Errorf("unexpected model: %s", cr.Model)
	}
	if !cr.Stream {
		t.Error("expected streaming to be true")
	}
	if len(cr.Messages) != 2 {
		t.Errorf("expected 2 messages (system + user), got %d", len(cr.Messages))
	}
	if cr.Messages[0]["role"] != "system" {
		t.Error("first message should be system")
	}
}

func TestEncodeRequest_MultimodalImageURL(t *testing.T) {
	msg := ai.Message{
		Role: ai.RoleUser,
		Content: []ai.ContentPart{
			ai.TextPart("what is this?"),
			ai.ImageURLPart("https://example.com/img.png"),
		},
	}
	req := ai.LanguageModelRequest{Messages: []ai.Message{msg}}
	cr, err := encodeRequest("gemini-2.5-flash", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	content, ok := cr.Messages[0]["content"].([]map[string]any)
	if !ok || len(content) != 2 {
		t.Errorf("expected multipart content with 2 parts, got %v", cr.Messages[0]["content"])
	}
}
