package openaichat_test

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

func streamFromString(s string) io.ReadCloser {
	return io.NopCloser(strings.NewReader(s))
}

func collectEvents(body io.ReadCloser, params openaichat.SSEDecodeParams) []ai.StreamEvent {
	ch := make(chan ai.StreamEvent, 128)
	openaichat.DecodeSSEStream(context.Background(), body, ch, params)
	var events []ai.StreamEvent
	for e := range ch {
		events = append(events, e)
	}
	return events
}

func defaultParams() openaichat.SSEDecodeParams {
	return openaichat.SSEDecodeParams{ProviderName: "test"}
}

func TestDecodeSSE_TextOnly(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}
data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
data: [DONE]
`
	events := collectEvents(streamFromString(sse), defaultParams())

	textDeltas, usageCount, finishCount := 0, 0, 0
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
	sse := `data: {"choices":[{"delta":{"content":"thinking...","thought":true},"finish_reason":null}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse), defaultParams())

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
	events := collectEvents(streamFromString(sse), defaultParams())

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
	cancel()

	sse := `data: {"choices":[{"delta":{"content":"text"}}]}
data: [DONE]
`
	ch := make(chan ai.StreamEvent, 128)
	openaichat.DecodeSSEStream(ctx, streamFromString(sse), ch, defaultParams())

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

func TestDecodeSSE_RawFinishReasonPopulated(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse), defaultParams())

	var finishEvents []ai.StreamEvent
	for _, e := range events {
		if e.Type == ai.StreamEventFinish {
			finishEvents = append(finishEvents, e)
		}
	}
	if len(finishEvents) == 0 {
		t.Fatal("expected at least one finish event")
	}

	found := false
	for _, e := range finishEvents {
		if e.RawFinishReason == "stop" {
			found = true
			if e.FinishReason != ai.FinishReasonStop {
				t.Errorf("expected FinishReasonStop, got %q", e.FinishReason)
			}
		}
	}
	if !found {
		t.Error("expected a finish event with RawFinishReason=stop")
	}
}

func TestDecodeSSE_ToolCallsRawFinishReason(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"search","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse), defaultParams())

	for _, e := range events {
		if e.Type == ai.StreamEventFinish && e.RawFinishReason == "tool_calls" {
			if e.FinishReason != ai.FinishReasonToolCalls {
				t.Errorf("expected FinishReasonToolCalls, got %q", e.FinishReason)
			}
			return
		}
	}
	t.Error("expected finish event with RawFinishReason=tool_calls")
}

func TestDecodeSSE_DoneRawFinishReason(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse), defaultParams())

	for _, e := range events {
		if e.Type == ai.StreamEventFinish {
			if e.RawFinishReason != "stop" {
				t.Errorf("expected RawFinishReason=stop from [DONE], got %q", e.RawFinishReason)
			}
			return
		}
	}
	t.Error("no finish event found")
}

func TestDecodeSSE_MetadataExtractor(t *testing.T) {
	sse := `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
`
	called := false
	params := openaichat.SSEDecodeParams{
		ProviderName: "test",
		MetadataExtractor: func(_ openaichat.StreamChunk) map[string]any {
			called = true
			return map[string]any{"custom": "value"}
		},
	}
	events := collectEvents(streamFromString(sse), params)

	if !called {
		t.Error("expected MetadataExtractor to be called")
	}
	for _, e := range events {
		if e.Type == ai.StreamEventFinish && e.RawFinishReason == "stop" {
			if e.ProviderMetadata["custom"] != "value" {
				t.Errorf("expected metadata custom=value, got %v", e.ProviderMetadata)
			}
			return
		}
	}
	t.Error("expected finish event with provider metadata")
}
