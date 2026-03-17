package engine

import (
	"context"
	"fmt"
	"testing"
)

type capturingModel struct {
	calls    [][]StreamEvent
	idx      int
	requests []Request
}

func (m *capturingModel) ModelID() string { return "capturing" }

func (m *capturingModel) Stream(_ context.Context, req Request) (<-chan StreamEvent, error) {
	m.requests = append(m.requests, req)
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

func TestPrepareStep_NilIsNoOp(t *testing.T) {
	model := &mockModel{calls: [][]StreamEvent{
		{textEvt("hello"), finishEvt(FinishReasonStop)},
	}}

	ch := Run(context.Background(), RunParams{
		Model:       model,
		MaxSteps:    5,
		PrepareStep: nil,
	})

	var text string
	for ev := range ch {
		if ev.Type == StepEventTextDelta {
			text += ev.TextDelta
		}
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}
	if text != "hello" {
		t.Errorf("expected 'hello', got %q", text)
	}
}

func TestPrepareStep_ModelOverride(t *testing.T) {
	mainModel := &capturingModel{calls: [][]StreamEvent{
		{toolCallEvt(0, "tc1", "search", `{}`), finishEvt(FinishReasonToolCalls)},
	}}
	altModel := &capturingModel{calls: [][]StreamEvent{
		{textEvt("from alt"), finishEvt(FinishReasonStop)},
	}}

	exec := &mockExecutor{}
	ch := Run(context.Background(), RunParams{
		Model: mainModel,
		Tools: &ToolSet{Executor: exec},
		PrepareStep: func(ctx PrepareStepContext) *PrepareStepResult {
			if ctx.StepNumber == 1 {
				return &PrepareStepResult{Model: altModel}
			}
			return nil
		},
		MaxSteps: 5,
	})

	var text string
	for ev := range ch {
		if ev.Type == StepEventTextDelta {
			text += ev.TextDelta
		}
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}
	if text != "from alt" {
		t.Errorf("expected 'from alt', got %q", text)
	}
	if len(altModel.requests) != 1 {
		t.Errorf("expected alt model to be called once, got %d", len(altModel.requests))
	}
}

func TestPrepareStep_SystemOverride(t *testing.T) {
	model := &capturingModel{calls: [][]StreamEvent{
		{toolCallEvt(0, "tc1", "search", `{}`), finishEvt(FinishReasonToolCalls)},
		{textEvt("done"), finishEvt(FinishReasonStop)},
	}}
	exec := &mockExecutor{}

	ch := Run(context.Background(), RunParams{
		Model:   model,
		Tools:   &ToolSet{Executor: exec},
		Request: Request{System: "original system"},
		PrepareStep: func(ctx PrepareStepContext) *PrepareStepResult {
			if ctx.StepNumber == 1 {
				return &PrepareStepResult{System: "step-1 system"}
			}
			return nil
		},
		MaxSteps: 5,
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if len(model.requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(model.requests))
	}
	if model.requests[0].System != "" {
		t.Errorf("step 0: system should be empty (already in history), got %q", model.requests[0].System)
	}
	if model.requests[1].System != "step-1 system" {
		t.Errorf("step 1: expected 'step-1 system', got %q", model.requests[1].System)
	}
}

func TestPrepareStep_ToolChoiceOverride(t *testing.T) {
	model := &capturingModel{calls: [][]StreamEvent{
		{toolCallEvt(0, "tc1", "search", `{}`), finishEvt(FinishReasonToolCalls)},
		{textEvt("done"), finishEvt(FinishReasonStop)},
	}}
	exec := &mockExecutor{}

	ch := Run(context.Background(), RunParams{
		Model: model,
		Tools: &ToolSet{Executor: exec},
		PrepareStep: func(ctx PrepareStepContext) *PrepareStepResult {
			if ctx.StepNumber == 1 {
				return &PrepareStepResult{ToolChoice: &ToolChoice{Type: "none"}}
			}
			return nil
		},
		MaxSteps: 5,
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if len(model.requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(model.requests))
	}
	if model.requests[1].ToolChoice == nil || model.requests[1].ToolChoice.Type != "none" {
		t.Errorf("step 1: expected ToolChoice{Type:none}, got %v", model.requests[1].ToolChoice)
	}
}

func TestPrepareStep_StepsAccumulated(t *testing.T) {
	model := &mockModel{calls: [][]StreamEvent{
		{toolCallEvt(0, "tc1", "search", `{}`), finishEvt(FinishReasonToolCalls)},
		{toolCallEvt(0, "tc2", "fetch", `{}`), finishEvt(FinishReasonToolCalls)},
		{textEvt("done"), finishEvt(FinishReasonStop)},
	}}
	exec := &mockExecutor{}

	var capturedSteps [][]StepResultInfo
	ch := Run(context.Background(), RunParams{
		Model: model,
		Tools: &ToolSet{Executor: exec},
		PrepareStep: func(ctx PrepareStepContext) *PrepareStepResult {
			copied := make([]StepResultInfo, len(ctx.Steps))
			copy(copied, ctx.Steps)
			capturedSteps = append(capturedSteps, copied)
			return nil
		},
		MaxSteps: 5,
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if len(capturedSteps) != 3 {
		t.Fatalf("expected 3 PrepareStep calls, got %d", len(capturedSteps))
	}
	if len(capturedSteps[0]) != 0 {
		t.Errorf("step 0 should have 0 completed steps, got %d", len(capturedSteps[0]))
	}
	if len(capturedSteps[1]) != 1 {
		t.Errorf("step 1 should have 1 completed step, got %d", len(capturedSteps[1]))
	}
	if !capturedSteps[1][0].HasToolCalls {
		t.Error("step 0 result should have HasToolCalls=true")
	}
	if len(capturedSteps[2]) != 2 {
		t.Errorf("step 2 should have 2 completed steps, got %d", len(capturedSteps[2]))
	}
}

func TestActiveTools_FiltersByName(t *testing.T) {
	model := &capturingModel{calls: [][]StreamEvent{
		{textEvt("done"), finishEvt(FinishReasonStop)},
	}}

	allTools := []ToolDefinition{
		{Name: "search", Description: "search the web"},
		{Name: "fetch", Description: "fetch a URL"},
		{Name: "write", Description: "write a file"},
	}

	ch := Run(context.Background(), RunParams{
		Model:   model,
		Request: Request{Tools: allTools},
		PrepareStep: func(ctx PrepareStepContext) *PrepareStepResult {
			return &PrepareStepResult{ActiveTools: []string{"search", "write"}}
		},
		MaxSteps: 5,
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if len(model.requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(model.requests))
	}
	tools := model.requests[0].Tools
	if len(tools) != 2 {
		t.Fatalf("expected 2 filtered tools, got %d", len(tools))
	}
	names := make(map[string]bool)
	for _, td := range tools {
		names[td.Name] = true
	}
	if !names["search"] || !names["write"] {
		t.Errorf("expected search and write, got %v", names)
	}
}

func TestActiveTools_NilMeansAllTools(t *testing.T) {
	model := &capturingModel{calls: [][]StreamEvent{
		{textEvt("done"), finishEvt(FinishReasonStop)},
	}}

	allTools := []ToolDefinition{
		{Name: "search"},
		{Name: "fetch"},
	}

	ch := Run(context.Background(), RunParams{
		Model:   model,
		Request: Request{Tools: allTools},
		PrepareStep: func(ctx PrepareStepContext) *PrepareStepResult {
			return nil
		},
		MaxSteps: 5,
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if len(model.requests[0].Tools) != 2 {
		t.Errorf("expected all 2 tools, got %d", len(model.requests[0].Tools))
	}
}

func TestActiveTools_EmptySliceClearsTools(t *testing.T) {
	model := &capturingModel{calls: [][]StreamEvent{
		{textEvt("done"), finishEvt(FinishReasonStop)},
	}}

	allTools := []ToolDefinition{
		{Name: "search"},
		{Name: "fetch"},
	}

	ch := Run(context.Background(), RunParams{
		Model:   model,
		Request: Request{Tools: allTools},
		PrepareStep: func(ctx PrepareStepContext) *PrepareStepResult {
			return &PrepareStepResult{ActiveTools: []string{}}
		},
		MaxSteps: 5,
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if len(model.requests[0].Tools) != 0 {
		t.Errorf("expected 0 tools with empty ActiveTools, got %d", len(model.requests[0].Tools))
	}
}

func TestLifecycleCallbacks_OnStepFinish(t *testing.T) {
	model := &mockModel{calls: [][]StreamEvent{
		{toolCallEvt(0, "tc1", "search", `{"q":"test"}`), finishEvt(FinishReasonToolCalls)},
		{textEvt("done"), finishEvt(FinishReasonStop)},
	}}
	exec := &mockExecutor{results: map[string]string{"search": `{"ok":true}`}}

	var stepEvents []StepFinishEvent
	ch := Run(context.Background(), RunParams{
		Model:    model,
		Tools:    &ToolSet{Executor: exec},
		MaxSteps: 5,
		Callbacks: &LifecycleCallbacks{
			OnStepFinish: func(ev StepFinishEvent) {
				stepEvents = append(stepEvents, ev)
			},
		},
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if len(stepEvents) != 2 {
		t.Fatalf("expected 2 OnStepFinish calls, got %d", len(stepEvents))
	}
	if stepEvents[0].StepNumber != 0 {
		t.Errorf("first step should be 0, got %d", stepEvents[0].StepNumber)
	}
	if len(stepEvents[0].ToolCalls) != 1 || stepEvents[0].ToolCalls[0].Name != "search" {
		t.Errorf("first step should have search tool call, got %v", stepEvents[0].ToolCalls)
	}
	if stepEvents[1].StepNumber != 1 {
		t.Errorf("second step should be 1, got %d", stepEvents[1].StepNumber)
	}
}

func TestLifecycleCallbacks_OnFinish(t *testing.T) {
	model := &mockModel{calls: [][]StreamEvent{
		{textEvt("hello world"), finishEvt(FinishReasonStop)},
	}}

	var finishEvent *FinishEvent
	ch := Run(context.Background(), RunParams{
		Model:    model,
		MaxSteps: 5,
		Callbacks: &LifecycleCallbacks{
			OnFinish: func(ev FinishEvent) {
				finishEvent = &ev
			},
		},
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if finishEvent == nil {
		t.Fatal("expected OnFinish to be called")
	}
	if finishEvent.Text != "hello world" {
		t.Errorf("expected text 'hello world', got %q", finishEvent.Text)
	}
	if finishEvent.FinishReason != FinishReasonStop {
		t.Errorf("expected FinishReasonStop, got %q", finishEvent.FinishReason)
	}
}

func TestLifecycleCallbacks_OnChunk(t *testing.T) {
	model := &mockModel{calls: [][]StreamEvent{
		{textEvt("hi"), finishEvt(FinishReasonStop)},
	}}

	var chunkCount int
	ch := Run(context.Background(), RunParams{
		Model:    model,
		MaxSteps: 5,
		Callbacks: &LifecycleCallbacks{
			OnChunk: func(ev StepEvent) {
				chunkCount++
			},
		},
	})

	for ev := range ch {
		if ev.Type == StepEventError {
			t.Fatalf("unexpected error: %v", ev.Error)
		}
	}

	if chunkCount == 0 {
		t.Error("expected OnChunk to be called at least once")
	}
}

func TestLifecycleCallbacks_OnError(t *testing.T) {
	model := &mockModel{calls: [][]StreamEvent{
		{{Type: StreamEventError, Error: errTest}},
	}}

	var gotError error
	ch := Run(context.Background(), RunParams{
		Model:    model,
		MaxSteps: 5,
		Callbacks: &LifecycleCallbacks{
			OnError: func(err error) {
				gotError = err
			},
		},
	})

	for range ch {
	}

	if gotError == nil {
		t.Fatal("expected OnError to be called")
	}
}

var errTest = fmt.Errorf("test error")
