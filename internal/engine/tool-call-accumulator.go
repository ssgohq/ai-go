package engine

import (
	"encoding/json"
	"sort"
)

// toolCallState accumulates streaming argument fragments for a single tool call.
type toolCallState struct {
	id               string
	name             string
	args             string
	thoughtSignature string
	hasFinished      bool // skip further deltas after JSON is complete
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
		s := &toolCallState{
			id:               ev.ToolCallID,
			name:             ev.ToolCallName,
			args:             ev.ToolCallArgsDelta,
			thoughtSignature: ev.ThoughtSignature,
		}
		if s.args != "" && json.Valid([]byte(s.args)) {
			s.hasFinished = true
		}
		a.states[ev.ToolCallIndex] = s
		return true
	}
	if ev.ThoughtSignature != "" && state.thoughtSignature == "" {
		state.thoughtSignature = ev.ThoughtSignature
	}
	if ev.ToolCallArgsDelta != "" && !state.hasFinished {
		state.args += ev.ToolCallArgsDelta
		if json.Valid([]byte(state.args)) {
			state.hasFinished = true
		}
	}
	return false
}

// completed returns all accumulated tool calls sorted by index.
func (a *toolCallAccumulator) completed() []toolCallState {
	if len(a.states) == 0 {
		return nil
	}
	// Collect indices and sort for deterministic order.
	indices := make([]int, 0, len(a.states))
	for idx := range a.states {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	out := make([]toolCallState, 0, len(a.states))
	for _, idx := range indices {
		out = append(out, *a.states[idx])
	}
	return out
}

func (a *toolCallAccumulator) hasToolCalls() bool { return len(a.states) > 0 }
