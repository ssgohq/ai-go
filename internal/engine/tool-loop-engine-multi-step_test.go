package engine

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// mockModel simulates a Model returning pre-canned event sequences.
type mockModel struct {
	calls [][]StreamEvent
	idx   int
}

func (m *mockModel) ModelID() string { return "mock" }

func (m *mockModel) Stream(_ context.Context, _ Request) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent, 32)
	events := m.calls[m.idx]
	m.idx++
	go func() {
		defer close(ch)
		for _, e := range events {
			ch <- e
		}
	}()
	return ch, nil
}

// mockExecutor records calls and returns fixed results.
type mockExecutor struct {
	results map[string]string
	called  []string
}

func (e *mockExecutor) Execute(_ context.Context, name, _ string) (string, error) {
	e.called = append(e.called, name)
	if r, ok := e.results[name]; ok {
		return r, nil
	}
	return `{"ok":true}`, nil
}

func textEvt(s string) StreamEvent {
	return StreamEvent{Type: StreamEventTextDelta, TextDelta: s}
}

func toolCallEvt(idx int, id, name, args string) StreamEvent {
	return StreamEvent{
		Type:              StreamEventToolCallDelta,
		ToolCallIndex:     idx,
		ToolCallID:        id,
		ToolCallName:      name,
		ToolCallArgsDelta: args,
	}
}

func finishEvt(r FinishReason) StreamEvent {
	return StreamEvent{Type: StreamEventFinish, FinishReason: r}
}

func TestRunLoop_TextOnly(t *testing.T) {
	model := &mockModel{calls: [][]StreamEvent{
		{textEvt("Hello "), textEvt("world"), finishEvt(FinishReasonStop)},
	}}

	ch := Run(context.Background(), RunParams{Model: model, MaxSteps: 5})

	var texts []string
	var gotDone bool
	for ev := range ch {
		switch ev.Type {
		case StepEventTextDelta:
			texts = append(texts, ev.TextDelta)
		case StepEventDone:
			gotDone = true
		case StepEventError:
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if strings.Join(texts, "") != "Hello world" {
		t.Errorf("unexpected text: %q", strings.Join(texts, ""))
	}
	if !gotDone {
		t.Error("expected StepEventDone")
	}
}

func TestRunLoop_SingleToolCall(t *testing.T) {
	exec := &mockExecutor{results: map[string]string{"get_time": `{"time":"12:00"}`}}

	model := &mockModel{calls: [][]StreamEvent{
		{toolCallEvt(0, "tc1", "get_time", `{"tz":"UTC"}`), finishEvt(FinishReasonToolCalls)},
		{textEvt("It is 12:00 UTC"), finishEvt(FinishReasonStop)},
	}}

	ch := Run(context.Background(), RunParams{
		Model:    model,
		Tools:    &ToolSet{Executor: exec},
		MaxSteps: 5,
	})

	var stepStarts, toolResults, doneCount int
	for ev := range ch {
		switch ev.Type {
		case StepEventStepStart:
			stepStarts++
		case StepEventToolResult:
			toolResults++
		case StepEventDone:
			doneCount++
		case StepEventError:
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if stepStarts != 2 {
		t.Errorf("expected 2 step starts, got %d", stepStarts)
	}
	if toolResults != 1 {
		t.Errorf("expected 1 tool result, got %d", toolResults)
	}
	if doneCount != 1 {
		t.Errorf("expected 1 done event, got %d", doneCount)
	}
	if len(exec.called) != 1 || exec.called[0] != "get_time" {
		t.Errorf("expected get_time to be called, got %v", exec.called)
	}
}

func TestRunLoop_StepCountIs(t *testing.T) {
	exec := &mockExecutor{}
	model := &mockModel{calls: [][]StreamEvent{
		{toolCallEvt(0, "tc1", "search", `{"q":"a"}`), finishEvt(FinishReasonToolCalls)},
		{toolCallEvt(0, "tc2", "search", `{"q":"b"}`), finishEvt(FinishReasonToolCalls)},
		{textEvt("done"), finishEvt(FinishReasonStop)},
	}}

	stopAfter1 := StopCondition(func(step int, _ *StepResult) bool { return step >= 1 })

	ch := Run(context.Background(), RunParams{
		Model:    model,
		Tools:    &ToolSet{Executor: exec},
		StopWhen: stopAfter1,
		MaxSteps: 10,
	})

	var stepEnds int
	for ev := range ch {
		if ev.Type == StepEventStepEnd {
			stepEnds++
		}
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if stepEnds != 1 {
		t.Errorf("expected 1 step end (stopped early), got %d", stepEnds)
	}
}

func TestRunLoop_MaxStepsExhausted(t *testing.T) {
	// When maxSteps is hit with pending tool_calls, the loop exits honestly
	// with the last step's finish reason (ToolCalls). No forced "final text"
	// pass is fired — matches ai-sdk-node semantics. Caller decides how to
	// continue (bump budget, call again with tool_choice=none, etc.).
	exec := &mockExecutor{}
	calls := make([][]StreamEvent, 3) // exactly 3 tool-call steps, nothing more
	for i := 0; i < 3; i++ {
		calls[i] = []StreamEvent{
			toolCallEvt(0, "tc", "loop", `{}`),
			finishEvt(FinishReasonToolCalls),
		}
	}
	model := &mockModel{calls: calls}

	ch := Run(context.Background(), RunParams{
		Model:    model,
		Tools:    &ToolSet{Executor: exec},
		MaxSteps: 3,
	})

	var doneCount, stepStarts int
	var lastFinish FinishReason
	for ev := range ch {
		switch ev.Type {
		case StepEventStepStart:
			stepStarts++
		case StepEventStepEnd:
			lastFinish = ev.FinishReason
		case StepEventDone:
			doneCount++
		case StepEventError:
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}
	if doneCount != 1 {
		t.Errorf("expected 1 done event, got %d", doneCount)
	}
	if stepStarts != 3 {
		t.Errorf("expected exactly 3 step starts (no forced final pass), got %d", stepStarts)
	}
	if lastFinish != FinishReasonToolCalls {
		t.Errorf("expected last step finish=ToolCalls (honest exit), got %v", lastFinish)
	}
}

// TestRunLoop_ToolsNeverStripped is the regression guard for the Harmony-leak
// family of bugs (Thai / Chinese / private-use-unicode garbage appearing in
// delta.content when the gateway loses tool schema context). Verifies that no
// model call inside the tool loop is issued with a smaller tools slice than
// what the caller supplied — neither during normal steps nor at maxSteps
// exhaustion. Matches ai-sdk-node behavior where every doGenerate call sees
// the same stepTools slice unless the caller explicitly filters via
// PrepareStep.ActiveTools.
func TestRunLoop_ToolsNeverStripped(t *testing.T) {
	exec := &mockExecutor{}
	// 2 tool-call steps, then one text step. No emitFinalGeneration should fire.
	calls := [][]StreamEvent{
		{toolCallEvt(0, "tc1", "search", `{}`), finishEvt(FinishReasonToolCalls)},
		{toolCallEvt(0, "tc2", "search", `{}`), finishEvt(FinishReasonToolCalls)},
		{textEvt("final answer"), finishEvt(FinishReasonStop)},
	}
	rm := &recordingModel{mockModel: mockModel{calls: calls}}

	inputTools := []ToolDefinition{
		{Name: "search"},
		{Name: "fetch"},
		{Name: "browse"},
	}

	ch := Run(context.Background(), RunParams{
		Model: rm,
		Request: Request{
			Tools:    inputTools,
			Messages: []Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "hi"}}}},
		},
		Tools:    &ToolSet{Definitions: inputTools, Executor: exec},
		MaxSteps: 5,
	})
	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if len(rm.requests) != 3 {
		t.Fatalf("expected 3 model calls (2 tool + 1 text), got %d", len(rm.requests))
	}
	for i, r := range rm.requests {
		if len(r.Tools) != len(inputTools) {
			t.Errorf("step %d: tools stripped — expected %d tools, got %d",
				i, len(inputTools), len(r.Tools))
		}
	}
}

// TestRunLoop_MaxStepsExhausted_OnFinishUsesLastStepSr verifies the OnFinish
// callback receives the actual last-step streamResult (FinishReasonToolCalls)
// instead of a faked FinishReasonStop that the old emitFinalGeneration path
// used to synthesize. Matches ai-sdk-node: honest signal about loop state.
func TestRunLoop_MaxStepsExhausted_OnFinishUsesLastStepSr(t *testing.T) {
	exec := &mockExecutor{}
	calls := [][]StreamEvent{
		{toolCallEvt(0, "tc", "loop", `{}`), finishEvt(FinishReasonToolCalls)},
		{toolCallEvt(0, "tc", "loop", `{}`), finishEvt(FinishReasonToolCalls)},
	}
	model := &mockModel{calls: calls}

	var finishEvent FinishEvent
	var finishSeen bool
	ch := Run(context.Background(), RunParams{
		Model:    model,
		Tools:    &ToolSet{Executor: exec},
		MaxSteps: 2,
		Callbacks: &LifecycleCallbacks{
			OnFinish: func(event FinishEvent) {
				finishEvent = event
				finishSeen = true
			},
		},
	})
	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if !finishSeen {
		t.Fatal("OnFinish was not called")
	}
	if finishEvent.FinishReason != FinishReasonToolCalls {
		t.Errorf("OnFinish.FinishReason: expected ToolCalls (honest), got %v", finishEvent.FinishReason)
	}
	if len(finishEvent.Steps) != 2 {
		t.Errorf("expected 2 completed steps, got %d", len(finishEvent.Steps))
	}
}

// recordingModel wraps mockModel and records each Request received.
type recordingModel struct {
	mockModel
	requests []Request
}

func (m *recordingModel) Stream(ctx context.Context, req Request) (<-chan StreamEvent, error) {
	m.requests = append(m.requests, req)
	return m.mockModel.Stream(ctx, req)
}

func TestParseStructuredOutput_ValidJSON(t *testing.T) {
	raw := `{"name":"Alice","age":30}`
	got := parseStructuredOutput(raw)
	if got == nil {
		t.Fatal("expected non-nil result")
	}
	var m map[string]any
	if err := json.Unmarshal(got, &m); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if m["name"] != "Alice" {
		t.Errorf("unexpected name: %v", m["name"])
	}
}

func TestParseStructuredOutput_FencedJSON(t *testing.T) {
	raw := "```json\n{\"ok\":true}\n```"
	got := parseStructuredOutput(raw)
	if got == nil {
		t.Fatal("expected non-nil result for fenced JSON")
	}
}

func TestParseStructuredOutput_InvalidJSON(t *testing.T) {
	got := parseStructuredOutput("not json at all")
	if got != nil {
		t.Error("expected nil for invalid JSON")
	}
}
