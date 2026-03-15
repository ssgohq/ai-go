package gemini

import (
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// TestGeminiContract_RawFinishReasonPopulated verifies that the Gemini SSE decoder
// populates RawFinishReason on StreamEventFinish events with the original provider string.
func TestGeminiContract_RawFinishReasonPopulated(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	var finishEvents []ai.StreamEvent
	for _, e := range events {
		if e.Type == ai.StreamEventFinish {
			finishEvents = append(finishEvents, e)
		}
	}

	// The choice-level finish_reason="stop" and the [DONE] sentinel both emit finish events.
	if len(finishEvents) == 0 {
		t.Fatal("expected at least one finish event")
	}

	// Find the finish event with raw reason "stop" from the choice delta.
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

// TestGeminiContract_DoneRawFinishReason verifies the [DONE] sentinel emits RawFinishReason="stop".
func TestGeminiContract_DoneRawFinishReason(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

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

// TestGeminiContract_ToolCallsRawFinishReason verifies tool_calls finish reason is passed through raw.
func TestGeminiContract_ToolCallsRawFinishReason(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"search","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

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
