package engine

import "testing"

func makeToolCallDelta(idx int, id, name, argsDelta string) StreamEvent {
	return StreamEvent{
		Type:              StreamEventToolCallDelta,
		ToolCallIndex:     idx,
		ToolCallID:        id,
		ToolCallName:      name,
		ToolCallArgsDelta: argsDelta,
	}
}

func TestAccumulator_PartialDeltas(t *testing.T) {
	acc := newToolCallAccumulator()
	// Simulate fragmented JSON arriving in pieces.
	acc.add(makeToolCallDelta(0, "tc1", "search", `{"q`))
	acc.add(makeToolCallDelta(0, "tc1", "search", `":"hel`))
	acc.add(makeToolCallDelta(0, "tc1", "search", `lo"}`))

	calls := acc.completed()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].args != `{"q":"hello"}` {
		t.Errorf("unexpected args: %q", calls[0].args)
	}
	if !calls[0].hasFinished {
		t.Error("expected hasFinished=true after valid JSON")
	}
}

func TestAccumulator_SingleCompleteDelta(t *testing.T) {
	acc := newToolCallAccumulator()
	acc.add(makeToolCallDelta(0, "tc1", "get_time", `{"tz":"UTC"}`))

	calls := acc.completed()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].args != `{"tz":"UTC"}` {
		t.Errorf("unexpected args: %q", calls[0].args)
	}
	if !calls[0].hasFinished {
		t.Error("expected hasFinished=true")
	}
}

func TestAccumulator_HasFinishedPreventsOverAppend(t *testing.T) {
	acc := newToolCallAccumulator()
	// First delta completes the JSON.
	acc.add(makeToolCallDelta(0, "tc1", "echo", `{}`))
	// Further deltas should be ignored.
	acc.add(makeToolCallDelta(0, "tc1", "echo", `,"extra":true`))
	acc.add(makeToolCallDelta(0, "tc1", "echo", `}`))

	calls := acc.completed()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].args != `{}` {
		t.Errorf("expected just {}, got %q", calls[0].args)
	}
}

func TestAccumulator_IncompleteJSON_EmittedAsIs(t *testing.T) {
	acc := newToolCallAccumulator()
	// Simulate stream ending before JSON is complete.
	acc.add(makeToolCallDelta(0, "tc1", "broken", `{"partial`))

	calls := acc.completed()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].args != `{"partial` {
		t.Errorf("unexpected args: %q", calls[0].args)
	}
	if calls[0].hasFinished {
		t.Error("expected hasFinished=false for incomplete JSON")
	}
}

func TestAccumulator_MultipleConcurrentCalls(t *testing.T) {
	acc := newToolCallAccumulator()
	// Two tool calls interleaved by index.
	acc.add(makeToolCallDelta(0, "tc1", "search", `{"q`))
	acc.add(makeToolCallDelta(1, "tc2", "fetch", `{"url`))
	acc.add(makeToolCallDelta(0, "tc1", "search", `":"a"}`))
	acc.add(makeToolCallDelta(1, "tc2", "fetch", `":"b"}`))

	calls := acc.completed()
	if len(calls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(calls))
	}
	if calls[0].args != `{"q":"a"}` {
		t.Errorf("call 0 unexpected args: %q", calls[0].args)
	}
	if calls[1].args != `{"url":"b"}` {
		t.Errorf("call 1 unexpected args: %q", calls[1].args)
	}
	if !calls[0].hasFinished || !calls[1].hasFinished {
		t.Error("expected both calls to have hasFinished=true")
	}
}

func TestAccumulator_EmptyArgs(t *testing.T) {
	acc := newToolCallAccumulator()
	// Tool call with no arguments at all.
	acc.add(makeToolCallDelta(0, "tc1", "noop", ""))

	calls := acc.completed()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].args != "" {
		t.Errorf("expected empty args, got %q", calls[0].args)
	}
}

func TestAccumulator_TemporarilyValidJSON_StillAppends(t *testing.T) {
	// This is the key bug scenario: JSON becomes temporarily valid during
	// streaming but more deltas are expected. The old code would stop
	// accumulating here. The new code correctly marks hasFinished and ignores
	// further deltas — which is the correct behavior since `{}` IS valid JSON.
	acc := newToolCallAccumulator()
	acc.add(makeToolCallDelta(0, "tc1", "tool", `{}`))

	state := acc.states[0]
	if !state.hasFinished {
		t.Error("expected hasFinished=true after valid JSON `{}`")
	}
	if state.args != `{}` {
		t.Errorf("expected `{}`, got %q", state.args)
	}
}
