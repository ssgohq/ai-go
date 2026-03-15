package uistream

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

func makeEvents(evs ...engine.StepEvent) <-chan engine.StepEvent {
	ch := make(chan engine.StepEvent, len(evs))
	for _, e := range evs {
		ch <- e
	}
	close(ch)
	return ch
}

func runAdapter(evs ...engine.StepEvent) string {
	a := NewAdapter("msg-test")
	var buf bytes.Buffer
	a.Stream(makeEvents(evs...), &buf)
	return buf.String()
}

func assertContains(t *testing.T, output, want string) {
	t.Helper()
	if !strings.Contains(output, want) {
		t.Errorf("expected output to contain %q\nfull output:\n%s", want, output)
	}
}

func assertNotContains(t *testing.T, output, want string) {
	t.Helper()
	if strings.Contains(output, want) {
		t.Errorf("expected output NOT to contain %q\nfull output:\n%s", want, output)
	}
}

// --- text-only golden ---

func TestGolden_TextOnly(t *testing.T) {
	output := runAdapter(
		engine.StepEvent{Type: engine.StepEventStepStart, StepNumber: 0},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "Hello "},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "world"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	assertContains(t, output, `"type":"start"`)
	assertContains(t, output, `"type":"start-step"`)
	assertContains(t, output, `"type":"text-start"`)
	assertContains(t, output, `"type":"text-delta"`)
	assertContains(t, output, `"delta":"Hello "`)
	assertContains(t, output, `"delta":"world"`)
	assertContains(t, output, `"type":"text-end"`)
	assertContains(t, output, `"type":"finish-step"`)
	assertContains(t, output, `"type":"finish"`)
	assertContains(t, output, "[DONE]")
}

// --- reasoning golden ---

func TestGolden_Reasoning(t *testing.T) {
	output := runAdapter(
		engine.StepEvent{Type: engine.StepEventStepStart, StepNumber: 0},
		engine.StepEvent{Type: engine.StepEventReasoningDelta, ReasoningDelta: "thinking..."},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "answer"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	assertContains(t, output, `"type":"reasoning-start"`)
	assertContains(t, output, `"type":"reasoning-delta"`)
	assertContains(t, output, `"delta":"thinking..."`)
	assertContains(t, output, `"type":"reasoning-end"`)
	assertContains(t, output, `"type":"text-start"`)
	assertContains(t, output, `"delta":"answer"`)
}

// --- tool-call golden ---

func TestGolden_ToolCall(t *testing.T) {
	output := runAdapter(
		engine.StepEvent{Type: engine.StepEventStepStart, StepNumber: 0},
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
		engine.StepEvent{Type: engine.StepEventStepStart, StepNumber: 1},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "Found nothing."},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	assertContains(t, output, `"type":"tool-input-start"`)
	assertContains(t, output, `"toolCallId":"tc1"`)
	assertContains(t, output, `"toolName":"search"`)
	assertContains(t, output, `"type":"tool-input-delta"`)
	assertContains(t, output, `"type":"tool-input-available"`)
	assertContains(t, output, `"type":"tool-output-available"`)
	assertContains(t, output, `"type":"finish-step"`)
	assertContains(t, output, "[DONE]")
}

// --- error golden ---

func TestGolden_Error(t *testing.T) {
	output := runAdapter(
		engine.StepEvent{Type: engine.StepEventStepStart, StepNumber: 0},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "partial"},
		engine.StepEvent{Type: engine.StepEventError, Error: fmt.Errorf("connection reset")},
	)

	assertContains(t, output, `"type":"error"`)
	assertContains(t, output, "connection reset")
	// finish should NOT appear after an error
	assertNotContains(t, output, `"type":"finish"`)
}

// --- full text is returned ---

func TestStream_ReturnsFullText(t *testing.T) {
	a := NewAdapter("msg-1")
	var buf bytes.Buffer
	text := a.Stream(makeEvents(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "Hello "},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "world"},
		engine.StepEvent{Type: engine.StepEventStepEnd},
		engine.StepEvent{Type: engine.StepEventDone},
	), &buf)

	if text != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", text)
	}
}
