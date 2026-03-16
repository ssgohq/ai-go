package uistream

import (
	"bytes"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// TestMergeStreamResult_BasicText verifies model stream events are written to the Writer.
func TestMergeStreamResult_BasicText(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "hi"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var buf bytes.Buffer
	wr := NewWriter(&buf)
	text := wr.MergeStreamResult(sr)

	output := buf.String()
	if text != "hi" {
		t.Errorf("expected text=%q, got %q", "hi", text)
	}
	assertContains(t, output, `"type":"text-delta"`)
	assertContains(t, output, `"delta":"hi"`)
	// MergeStreamResult does NOT emit start or finish; caller manages lifecycle.
	assertNotContains(t, output, `"type":"finish"`)
}

// TestMergeStreamResult_CustomDataInterleaving verifies the full custom data + model
// stream interleaving pattern works correctly.
func TestMergeStreamResult_CustomDataInterleaving(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "answer"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var buf bytes.Buffer
	wr := NewWriter(&buf)

	wr.WriteStart("msg-merge")
	wr.WriteData("plan", map[string]string{"step": "1"})
	text := wr.MergeStreamResult(sr)
	wr.WriteData("sources", []string{"https://example.com"})
	wr.WriteFinish()

	output := buf.String()

	if text != "answer" {
		t.Errorf("expected merged text=%q, got %q", "answer", text)
	}
	// start comes before plan
	startIdx := strings.Index(output, `"type":"start"`)
	planIdx := strings.Index(output, `"type":"data-plan"`)
	textIdx := strings.Index(output, `"type":"text-delta"`)
	sourcesIdx := strings.Index(output, `"type":"data-sources"`)
	finishIdx := strings.Index(output, `"type":"finish"`)

	if startIdx < 0 {
		t.Error("missing start chunk")
	}
	if planIdx < 0 {
		t.Error("missing data-plan chunk")
	}
	if textIdx < 0 {
		t.Error("missing text-delta chunk")
	}
	if sourcesIdx < 0 {
		t.Error("missing data-sources chunk")
	}
	if finishIdx < 0 {
		t.Error("missing finish chunk")
	}

	// Order: start < plan < text < sources < finish
	if startIdx >= planIdx {
		t.Error("start should appear before data-plan")
	}
	if planIdx >= textIdx {
		t.Error("data-plan should appear before text-delta")
	}
	if textIdx >= sourcesIdx {
		t.Error("text-delta should appear before data-sources")
	}
	if sourcesIdx >= finishIdx {
		t.Error("data-sources should appear before finish")
	}
}

// TestMergeStreamResult_ToolResultHook verifies the hook fires during merge.
func TestMergeStreamResult_ToolResultHook(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{
			Type:              engine.StepEventToolCallStart,
			ToolCallID:        "tc2",
			ToolCallName:      "lookup",
			ToolCallArgsDelta: `{"key":"val"}`,
		},
		engine.StepEvent{
			Type: engine.StepEventToolResult,
			ToolResult: &engine.ToolResult{
				ID:     "tc2",
				Name:   "lookup",
				Args:   `{"key":"val"}`,
				Output: `"found"`,
			},
		},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonToolCalls},
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "result"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var hookFired bool
	var capturedID string
	hook := func(_ *Writer, result ToolResult) {
		hookFired = true
		capturedID = result.ToolCallID
	}

	var buf bytes.Buffer
	wr := NewWriter(&buf)
	wr.MergeStreamResult(sr, MergeWithToolResultHook(hook))

	if !hookFired {
		t.Error("expected tool result hook to fire")
	}
	if capturedID != "tc2" {
		t.Errorf("expected ToolCallID=tc2, got %q", capturedID)
	}
}

// TestMergeStreamResult_OnFinish verifies the on-finish callback fires.
func TestMergeStreamResult_OnFinish(t *testing.T) {
	sr := makeStreamResult(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "done"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var finishedText string
	var buf bytes.Buffer
	wr := NewWriter(&buf)
	wr.MergeStreamResult(sr, MergeWithOnFinish(func(text string) {
		finishedText = text
	}))

	if finishedText != "done" {
		t.Errorf("expected on-finish text=%q, got %q", "done", finishedText)
	}
}

// TestMergeStreamResult_ImplementsStreamEventer verifies *ai.StreamResult satisfies
// the StreamEventer interface used by MergeStreamResult.
func TestMergeStreamResult_ImplementsStreamEventer(t *testing.T) {
	var _ StreamEventer = (*ai.StreamResult)(nil)
}
