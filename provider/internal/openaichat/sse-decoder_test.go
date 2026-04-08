package openaichat

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// sseBody returns a ReadCloser from raw SSE lines.
func sseBody(lines ...string) io.ReadCloser {
	return io.NopCloser(strings.NewReader(strings.Join(lines, "\n") + "\n"))
}

// collectEvents drains a channel into a slice.
func collectEvents(ch <-chan ai.StreamEvent) []ai.StreamEvent {
	var events []ai.StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}
	return events
}

func TestSSE_ToolCallsFinishReason_NotOverwrittenByDone(t *testing.T) {
	// Simulate: chunk with finish_reason "tool_calls", then [DONE].
	body := sseBody(
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"get_weather","arguments":"{\"city\":\"HN\"}"}}]},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	)

	ch := make(chan ai.StreamEvent, 32)
	go DecodeSSEStream(context.Background(), body, ch, SSEDecodeParams{ProviderName: "test"})

	events := collectEvents(ch)

	var finishEvents []ai.StreamEvent
	for _, ev := range events {
		if ev.Type == ai.StreamEventFinish {
			finishEvents = append(finishEvents, ev)
		}
	}

	if len(finishEvents) != 1 {
		t.Fatalf("expected exactly 1 finish event, got %d", len(finishEvents))
	}
	if finishEvents[0].FinishReason != ai.FinishReasonToolCalls {
		t.Errorf("expected finish reason %q, got %q", ai.FinishReasonToolCalls, finishEvents[0].FinishReason)
	}
	if finishEvents[0].RawFinishReason != "tool_calls" {
		t.Errorf("expected raw finish reason %q, got %q", "tool_calls", finishEvents[0].RawFinishReason)
	}
}

func TestSSE_NoFinishReason_FallbackOnDone(t *testing.T) {
	// Simulate: text chunk without finish_reason, then [DONE] → fallback stop.
	body := sseBody(
		`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
		`data: [DONE]`,
	)

	ch := make(chan ai.StreamEvent, 32)
	go DecodeSSEStream(context.Background(), body, ch, SSEDecodeParams{ProviderName: "test"})

	events := collectEvents(ch)

	var finishEvents []ai.StreamEvent
	for _, ev := range events {
		if ev.Type == ai.StreamEventFinish {
			finishEvents = append(finishEvents, ev)
		}
	}

	if len(finishEvents) != 1 {
		t.Fatalf("expected exactly 1 finish event, got %d", len(finishEvents))
	}
	if finishEvents[0].FinishReason != ai.FinishReasonStop {
		t.Errorf("expected fallback finish reason %q, got %q", ai.FinishReasonStop, finishEvents[0].FinishReason)
	}
}

func TestSSE_StopFinishReason_NoDuplicate(t *testing.T) {
	// Simulate: chunk with finish_reason "stop", then [DONE] → only one finish event.
	body := sseBody(
		`data: {"choices":[{"delta":{"content":"Done"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	)

	ch := make(chan ai.StreamEvent, 32)
	go DecodeSSEStream(context.Background(), body, ch, SSEDecodeParams{ProviderName: "test"})

	events := collectEvents(ch)

	var finishEvents []ai.StreamEvent
	for _, ev := range events {
		if ev.Type == ai.StreamEventFinish {
			finishEvents = append(finishEvents, ev)
		}
	}

	if len(finishEvents) != 1 {
		t.Fatalf("expected exactly 1 finish event, got %d", len(finishEvents))
	}
	if finishEvents[0].FinishReason != ai.FinishReasonStop {
		t.Errorf("expected finish reason %q, got %q", ai.FinishReasonStop, finishEvents[0].FinishReason)
	}
}
