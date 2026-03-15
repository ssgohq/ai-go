package engine

import (
	"context"
	"testing"
)

func runStructuredOutputEngine(t *testing.T, modelResponse string) []StepEvent {
	t.Helper()

	model := &mockModel{calls: [][]StreamEvent{
		// Step 1: plain finish
		{
			{Type: StreamEventTextDelta, TextDelta: "analysis complete"},
			{Type: StreamEventFinish, FinishReason: FinishReasonStop},
		},
		// Structured output call
		{
			{Type: StreamEventTextDelta, TextDelta: modelResponse},
			{Type: StreamEventFinish, FinishReason: FinishReasonStop},
		},
	}}

	ch := Run(context.Background(), RunParams{
		Model: model,
		Request: Request{
			Output: &OutputSchema{
				Type:   "object",
				Schema: map[string]any{"type": "object"},
			},
		},
		MaxSteps: 5,
	})

	var events []StepEvent
	for ev := range ch {
		events = append(events, ev)
	}
	return events
}

func findStructuredOutput(events []StepEvent) (StepEvent, bool) {
	for _, ev := range events {
		if ev.Type == StepEventStructuredOutput {
			return ev, true
		}
	}
	return StepEvent{}, false
}

func TestStructuredOutput_ValidJSON(t *testing.T) {
	events := runStructuredOutputEngine(t, `{"score":42,"label":"good"}`)
	ev, ok := findStructuredOutput(events)
	if !ok {
		t.Fatal("expected StepEventStructuredOutput")
	}
	if string(ev.StructuredOutput) != `{"score":42,"label":"good"}` {
		t.Errorf("unexpected structured output: %s", ev.StructuredOutput)
	}
}

func TestStructuredOutput_FencedJSON(t *testing.T) {
	fenced := "```json\n{\"score\":7}\n```"
	events := runStructuredOutputEngine(t, fenced)
	ev, ok := findStructuredOutput(events)
	if !ok {
		t.Fatal("expected StepEventStructuredOutput for fenced JSON")
	}
	if string(ev.StructuredOutput) != `{"score":7}` {
		t.Errorf("unexpected structured output: %s", ev.StructuredOutput)
	}
}

func TestStructuredOutput_InvalidJSON(t *testing.T) {
	events := runStructuredOutputEngine(t, "not valid json at all")
	_, ok := findStructuredOutput(events)
	if ok {
		t.Error("expected no StepEventStructuredOutput for invalid JSON")
	}
}
