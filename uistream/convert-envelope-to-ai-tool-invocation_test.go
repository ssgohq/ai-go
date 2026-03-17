package uistream

import (
	"encoding/json"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// TestToAIContentParts_ToolInvocation_Call verifies state="call" produces a single ToolCallPart.
func TestToAIContentParts_ToolInvocation_Call(t *testing.T) {
	args := json.RawMessage(`{"query":"hello"}`)
	parts := []EnvelopePartUnion{
		{
			Type:       EnvelopePartTypeToolInvocation,
			ToolCallID: "tc-1",
			ToolName:   "search",
			Input:      args,
			State:      "call",
		},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	if got[0].Type != ai.ContentPartTypeToolCall {
		t.Errorf("expected tool_call type, got %q", got[0].Type)
	}
	if got[0].ToolCallID != "tc-1" {
		t.Errorf("expected ToolCallID=tc-1, got %q", got[0].ToolCallID)
	}
	if got[0].ToolCallName != "search" {
		t.Errorf("expected ToolCallName=search, got %q", got[0].ToolCallName)
	}
}

// TestToAIContentParts_ToolInvocation_PartialCall verifies state="partial-call" also emits a ToolCallPart.
func TestToAIContentParts_ToolInvocation_PartialCall(t *testing.T) {
	parts := []EnvelopePartUnion{
		{
			Type:       EnvelopePartTypeToolInvocation,
			ToolCallID: "tc-2",
			ToolName:   "calculator",
			State:      "partial-call",
		},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	if got[0].Type != ai.ContentPartTypeToolCall {
		t.Errorf("expected tool_call type for partial-call, got %q", got[0].Type)
	}
}

// TestToAIContentParts_ToolInvocation_Result verifies state="result" emits ToolCallPart + ToolResultPart.
func TestToAIContentParts_ToolInvocation_Result(t *testing.T) {
	args := json.RawMessage(`{"x":42}`)
	parts := []EnvelopePartUnion{
		{
			Type:       EnvelopePartTypeToolInvocation,
			ToolCallID: "tc-3",
			ToolName:   "add",
			Input:      args,
			Output:     "84",
			State:      "result",
		},
	}
	got := ToAIContentParts(parts)
	if len(got) != 2 {
		t.Fatalf("expected 2 parts (tool-call + tool-result), got %d", len(got))
	}
	if got[0].Type != ai.ContentPartTypeToolCall {
		t.Errorf("expected tool_call first, got %q", got[0].Type)
	}
	if got[1].Type != ai.ContentPartTypeToolResult {
		t.Errorf("expected tool_result second, got %q", got[1].Type)
	}
	if got[1].ToolResultID != "tc-3" {
		t.Errorf("expected ToolResultID=tc-3, got %q", got[1].ToolResultID)
	}
	if got[1].ToolResultOutput != "84" {
		t.Errorf("expected ToolResultOutput=84, got %q", got[1].ToolResultOutput)
	}
}

// TestToAIContentParts_ToolInvocation_UnknownStateSkipped verifies unknown states produce no parts.
func TestToAIContentParts_ToolInvocation_UnknownStateSkipped(t *testing.T) {
	parts := []EnvelopePartUnion{
		{
			Type:       EnvelopePartTypeToolInvocation,
			ToolCallID: "tc-4",
			ToolName:   "fn",
			State:      "error",
		},
	}
	got := ToAIContentParts(parts)
	if len(got) != 0 {
		t.Errorf("expected no parts for state=error, got %d", len(got))
	}
}

// TestToAIContentParts_ToolInvocation_JSONDecode verifies tool-invocation parts decode correctly from JSON.
func TestToAIContentParts_ToolInvocation_JSONDecode(t *testing.T) {
	raw := `{"type":"tool-invocation","toolCallId":"tc-5","toolName":"weather","input":{"city":"NYC"},"output":"sunny","state":"result"}`
	var p EnvelopePartUnion
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if p.Type != EnvelopePartTypeToolInvocation {
		t.Errorf("Type: got %q, want tool-invocation", p.Type)
	}
	if p.ToolCallID != "tc-5" {
		t.Errorf("ToolCallID: got %q", p.ToolCallID)
	}
	if p.ToolName != "weather" {
		t.Errorf("ToolName: got %q", p.ToolName)
	}
	if p.State != "result" {
		t.Errorf("State: got %q", p.State)
	}
	if p.Output != "sunny" {
		t.Errorf("Output: got %q", p.Output)
	}
}
