package uistream

import (
	"bytes"
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

// TestStreamToWriter_BasicTextStream verifies SSE output contains expected chunks.
func TestStreamToWriter_BasicTextStream(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "Hello "},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "world"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var buf bytes.Buffer
	text := StreamToWriter(sr, &buf, "msg-1")
	output := buf.String()

	if text != "Hello world" {
		t.Errorf("expected text=%q, got %q", "Hello world", text)
	}
	assertContains(t, output, `"type":"start"`)
	assertContains(t, output, `"messageId":"msg-1"`)
	assertContains(t, output, `"type":"text-delta"`)
	assertContains(t, output, `"delta":"Hello "`)
	assertContains(t, output, `"delta":"world"`)
	assertContains(t, output, `"type":"finish"`)
	assertContains(t, output, "[DONE]")
}

// TestStreamToWriter_ToolResultHookFires verifies the tool result hook is invoked.
func TestStreamToWriter_ToolResultHookFires(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{
			Type:              engine.StepEventToolCallStart,
			ToolCallID:        "tc1",
			ToolCallName:      "search",
			ToolCallArgsDelta: `{"q":"go"}`,
		},
		engine.StepEvent{
			Type: engine.StepEventToolResult,
			ToolResult: &engine.ToolResult{
				ID:     "tc1",
				Name:   "search",
				Args:   `{"q":"go"}`,
				Output: `{"results":[]}`,
			},
		},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonToolCalls},
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "done"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var hookFired bool
	var capturedResult ToolResult
	hook := func(_ *Writer, result ToolResult) {
		hookFired = true
		capturedResult = result
	}

	var buf bytes.Buffer
	StreamToWriter(sr, &buf, "msg-hook", WithUIToolResultHook(hook))

	if !hookFired {
		t.Error("expected tool result hook to fire")
	}
	if capturedResult.ToolCallID != "tc1" {
		t.Errorf("expected ToolCallID=tc1, got %q", capturedResult.ToolCallID)
	}
	if capturedResult.ToolName != "search" {
		t.Errorf("expected ToolName=search, got %q", capturedResult.ToolName)
	}
	if capturedResult.ArgsJSON != `{"q":"go"}` {
		t.Errorf("expected ArgsJSON=%q, got %q", `{"q":"go"}`, capturedResult.ArgsJSON)
	}
}

// TestStreamToWriter_OnFinishCallback verifies the onFinish callback is invoked with full text.
func TestStreamToWriter_OnFinishCallback(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "hello"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var finished string
	var buf bytes.Buffer
	StreamToWriter(sr, &buf, "msg-finish", WithUIOnFinish(func(text string) {
		finished = text
	}))

	if finished != "hello" {
		t.Errorf("expected onFinish text=%q, got %q", "hello", finished)
	}
}

// TestStreamToWriter_SSELineFormat verifies every line is prefixed with "data: ".
func TestStreamToWriter_SSELineFormat(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "x"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var buf bytes.Buffer
	StreamToWriter(sr, &buf, "msg-fmt")

	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			t.Errorf("SSE line missing 'data: ' prefix: %q", line)
		}
	}
}
