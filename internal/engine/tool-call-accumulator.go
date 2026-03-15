package engine

import "encoding/json"

// toolCallState accumulates streaming argument fragments for a single tool call.
type toolCallState struct {
	id               string
	name             string
	args             string
	thoughtSignature string
}

// toolCallAccumulator groups streaming tool-call deltas by index.
type toolCallAccumulator struct {
	states map[int]*toolCallState
}

func newToolCallAccumulator() *toolCallAccumulator {
	return &toolCallAccumulator{states: make(map[int]*toolCallState)}
}

// add integrates a StreamEvent tool-call delta. Returns true if this is a new index.
func (a *toolCallAccumulator) add(ev StreamEvent) bool {
	state, exists := a.states[ev.ToolCallIndex]
	if !exists {
		a.states[ev.ToolCallIndex] = &toolCallState{
			id:               ev.ToolCallID,
			name:             ev.ToolCallName,
			args:             ev.ToolCallArgsDelta,
			thoughtSignature: ev.ThoughtSignature,
		}
		return true
	}
	if ev.ThoughtSignature != "" && state.thoughtSignature == "" {
		state.thoughtSignature = ev.ThoughtSignature
	}
	if ev.ToolCallArgsDelta != "" && !json.Valid([]byte(state.args)) {
		state.args += ev.ToolCallArgsDelta
	}
	return false
}

// completed returns all accumulated tool calls in index order.
func (a *toolCallAccumulator) completed() []toolCallState {
	if len(a.states) == 0 {
		return nil
	}
	out := make([]toolCallState, 0, len(a.states))
	for i := 0; i < len(a.states); i++ {
		if s, ok := a.states[i]; ok {
			out = append(out, *s)
		}
	}
	return out
}

func (a *toolCallAccumulator) hasToolCalls() bool { return len(a.states) > 0 }
