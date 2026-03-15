package engine

import (
	"context"
	"testing"
)

// modelReturningJSON simulates a model that emits a structured JSON response.
type modelReturningJSON struct {
	response string
}

func (m *modelReturningJSON) ModelID() string { return "mock-structured" }

func (m *modelReturningJSON) Stream(_ context.Context, _ Request) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent, 4)
	go func() {
		defer close(ch)
		ch <- StreamEvent{Type: StreamEventTextDelta, TextDelta: m.response}
		ch <- StreamEvent{Type: StreamEventFinish, FinishReason: FinishReasonStop}
	}()
	return ch, nil
}

func runStructuredOutputEngine(t *testing.T, modelResponse string) []StepEvent {
	t.Helper()

	// Two model calls: first returns no tool calls (triggers structured output call),
	// second returns the JSON response.
	step1Model := &mockModel{calls: [][]StreamEvent{
		// Step 1: plain text, no tools → triggers structured output call
		{{Type: StreamEventTextDelta, TextDelta: "done"}, {Type: StreamEventFinish, FinishReason: FinishReasonStop}},
	}}

	// We need to intercept the structured output call. Use a two-call mockModel.
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
	_ = step1Model

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
