package uistream

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// toolResultEvent builds a StepEventToolResult with optional provider metadata.
func toolResultEvent(id, name, args, output string, pm map[string]any) engine.StepEvent {
	return engine.StepEvent{
		Type: engine.StepEventToolResult,
		ToolResult: &engine.ToolResult{
			ID:     id,
			Name:   name,
			Args:   args,
			Output: output,
		},
		ProviderMetadata: pm,
	}
}

// TestIntegration_MultiStepToolLoop_ChunkSequence verifies a 2-step tool-loop produces
// the expected chunk sequence: start, step events, tool chunks, text, finish.
func TestIntegration_MultiStepToolLoop_ChunkSequence(t *testing.T) {
	sr := newMockStreamEventer(
		// Step 1: model calls tool_A, gets result
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventToolCallStart, ToolCallID: "tc-1", ToolCallName: "search"},
		toolResultEvent("tc-1", "search", `{"q":"go"}`, `["result1"]`, nil),
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonToolCalls},

		// Step 2: model produces final answer
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "The answer is 42."},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-loop-1", ToUIStreamOptions{SendReasoning: true, SendSources: true})
	chunks := drainChunks(ch)

	// Verify lifecycle bookends.
	if _, ok := findChunk(chunks, ChunkStart); !ok {
		t.Error("expected start chunk")
	}
	if _, ok := findChunk(chunks, ChunkFinish); !ok {
		t.Error("expected finish chunk")
	}

	// Verify tool flow chunks for step 1.
	if _, ok := findChunk(chunks, ChunkToolInputStart); !ok {
		t.Error("expected tool-input-start chunk")
	}
	if _, ok := findChunk(chunks, ChunkToolInputAvailable); !ok {
		t.Error("expected tool-input-available chunk")
	}
	if _, ok := findChunk(chunks, ChunkToolOutputAvailable); !ok {
		t.Error("expected tool-output-available chunk")
	}

	// Verify text delta chunk in step 2.
	deltas := collectChunks(chunks, ChunkTextDelta)
	if len(deltas) == 0 {
		t.Error("expected text-delta chunks in step 2")
	}

	// Verify finish reason propagated.
	finish, _ := findChunk(chunks, ChunkFinish)
	if fr, _ := finish.Fields["finishReason"].(string); fr != "stop" {
		t.Errorf("expected finishReason=stop, got %q", fr)
	}
}

// TestIntegration_MultiStepToolLoop_PersistedParts verifies PersistedMessageBuilder
// accumulates tool-invocation and text parts correctly from a tool-loop stream.
func TestIntegration_MultiStepToolLoop_PersistedParts(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventToolCallStart, ToolCallID: "tc-2", ToolCallName: "lookup"},
		toolResultEvent("tc-2", "lookup", `{"key":"x"}`, `"found"`, nil),
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonToolCalls},

		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "Done."},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-loop-2", ToUIStreamOptions{SendReasoning: true, SendSources: true})

	builder := NewPersistedMessageBuilder()
	for c := range ch {
		builder.ObserveChunk(c)
	}

	parts := builder.Parts()
	if parts == nil {
		t.Fatal("expected non-nil parts")
	}

	var parsed []map[string]any
	if err := json.Unmarshal(parts, &parsed); err != nil {
		t.Fatalf("parts JSON unmarshal: %v", err)
	}

	// Should have tool-invocation + text parts.
	types := make([]string, 0, len(parsed))
	for _, p := range parsed {
		if typ, ok := p["type"].(string); ok {
			types = append(types, typ)
		}
	}

	hasToolInvocation := false
	hasText := false
	for _, typ := range types {
		if typ == "tool-invocation" {
			hasToolInvocation = true
		}
		if typ == "text" {
			hasText = true
		}
	}
	if !hasToolInvocation {
		t.Errorf("expected tool-invocation part, got types: %v", types)
	}
	if !hasText {
		t.Errorf("expected text part, got types: %v", types)
	}

	if content := builder.Content(); content != "Done." {
		t.Errorf("expected Content()=Done., got %q", content)
	}
}

// TestIntegration_ToolOutputError_InPersistedParts verifies that a manual WriteToolOutputError
// followed by observation in PersistedMessageBuilder produces a tool-invocation with state=error.
func TestIntegration_ToolOutputError_InPersistedParts(t *testing.T) {
	// Simulate: tool-input-available (happy path start), then tool-output-error.
	chunks := []Chunk{
		{Type: ChunkToolInputAvailable, Fields: map[string]any{
			"toolCallId": "tc-err",
			"toolName":   "flaky_tool",
			"input":      map[string]any{"n": 1},
		}},
		{Type: ChunkToolOutputError, Fields: map[string]any{
			"toolCallId": "tc-err",
			"errorText":  "connection timeout",
		}},
	}

	builder := NewPersistedMessageBuilder()
	for _, c := range chunks {
		builder.ObserveChunk(c)
	}

	parts := builder.Parts()
	if parts == nil {
		t.Fatal("expected non-nil parts after tool error")
	}

	var parsed []map[string]any
	if err := json.Unmarshal(parts, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(parsed) == 0 {
		t.Fatal("expected at least one persisted part")
	}

	// The tool-output-error should finalize as a tool-invocation with state=error.
	found := false
	for _, p := range parsed {
		if p["type"] == "tool-invocation" && p["state"] == "error" {
			found = true
			if et, _ := p["errorText"].(string); !strings.Contains(et, "timeout") {
				t.Errorf("expected errorText to contain 'timeout', got %q", et)
			}
		}
	}
	if !found {
		t.Errorf("expected tool-invocation part with state=error, got: %v", parsed)
	}
}

// TestIntegration_ToolOutputDenied_InPersistedParts verifies WriteToolOutputDenied
// produces a tool-invocation part with state=denied.
func TestIntegration_ToolOutputDenied_InPersistedParts(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkToolInputAvailable, Fields: map[string]any{
			"toolCallId": "tc-deny",
			"toolName":   "delete_file",
			"input":      map[string]any{"path": "/important"},
		}},
		{Type: ChunkToolOutputDenied, Fields: map[string]any{
			"toolCallId": "tc-deny",
		}},
	}

	builder := NewPersistedMessageBuilder()
	for _, c := range chunks {
		builder.ObserveChunk(c)
	}

	parts := builder.Parts()
	if parts == nil {
		t.Fatal("expected non-nil parts after denial")
	}

	var parsed []map[string]any
	if err := json.Unmarshal(parts, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	found := false
	for _, p := range parsed {
		if p["type"] == "tool-invocation" && p["state"] == "denied" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected tool-invocation with state=denied, got: %v", parsed)
	}
}
