package uistream

import (
	"encoding/json"
	"testing"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// TestIntegration_PersistedMessageBuilder_FullConversation verifies that a realistic
// chunk sequence (text + reasoning + tool + metadata) produces the correct Parts array.
func TestIntegration_PersistedMessageBuilder_FullConversation(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-full-1"}},
		{Type: ChunkStartStep, Fields: nil},

		// Reasoning block.
		{Type: ChunkReasoningStart, Fields: map[string]any{"id": "text_1"}},
		{Type: ChunkReasoningDelta, Fields: map[string]any{"id": "text_1", "delta": "Let me think..."}},
		{Type: ChunkReasoningEnd, Fields: map[string]any{"id": "text_1", "signature": "sig-abc"}},

		// Text block.
		{Type: ChunkTextStart, Fields: map[string]any{"id": "text_1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "text_1", "delta": "Here is my answer."}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "text_1"}},

		// Tool invocation (happy path).
		{Type: ChunkToolInputAvailable, Fields: map[string]any{
			"toolCallId": "tc-full-1",
			"toolName":   "calculator",
			"input":      map[string]any{"a": 2, "b": 3},
		}},
		{Type: ChunkToolOutputAvailable, Fields: map[string]any{
			"toolCallId": "tc-full-1",
			"output":     "5",
		}},

		// Message metadata chunk (emitted before finish in the real pipeline).
		{Type: ChunkMessageMetadata, Fields: map[string]any{
			"messageMetadata": map[string]any{"model": "gpt-4o", "totalTokens": 100},
		}},
		{Type: ChunkFinish, Fields: map[string]any{"finishReason": "stop"}},
	}

	builder := NewPersistedMessageBuilder()
	for _, c := range chunks {
		builder.ObserveChunk(c)
	}

	// Verify Content() returns concatenated text.
	if content := builder.Content(); content != "Here is my answer." {
		t.Errorf("Content(): got %q, want %q", content, "Here is my answer.")
	}

	// Verify Parts() contains expected part types.
	raw := builder.Parts()
	if raw == nil {
		t.Fatal("Parts() returned nil")
	}
	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err != nil {
		t.Fatalf("Parts() JSON unmarshal: %v", err)
	}

	typeSet := make(map[string]bool)
	for _, p := range parts {
		if typ, ok := p["type"].(string); ok {
			typeSet[typ] = true
		}
	}

	for _, want := range []string{"reasoning", "text", "tool-invocation"} {
		if !typeSet[want] {
			t.Errorf("expected part type %q in parts, got types: %v", want, typeSet)
		}
	}

	// Verify reasoning has signature.
	for _, p := range parts {
		if p["type"] == "reasoning" {
			if sig, ok := p["signature"].(string); !ok || sig != "sig-abc" {
				t.Errorf("reasoning part: expected signature=sig-abc, got %v", p["signature"])
			}
		}
	}

	// Verify tool-invocation has state=output-available and correct toolName.
	for _, p := range parts {
		if p["type"] == "tool-invocation" {
			if p["state"] != "output-available" {
				t.Errorf("tool-invocation state: got %v, want output-available", p["state"])
			}
			if p["toolName"] != "calculator" {
				t.Errorf("tool-invocation toolName: got %v, want calculator", p["toolName"])
			}
		}
	}

	// Verify Metadata() is set from the finish chunk.
	meta := builder.Metadata()
	if meta == nil {
		t.Fatal("Metadata() returned nil, expected message metadata from finish chunk")
	}
	var metaMap map[string]any
	if err := json.Unmarshal(meta, &metaMap); err != nil {
		t.Fatalf("Metadata() JSON unmarshal: %v", err)
	}
	if metaMap["model"] != "gpt-4o" {
		t.Errorf("Metadata model: got %v, want gpt-4o", metaMap["model"])
	}
}

// TestIntegration_EnvelopeRoundTrip_WithAllPartTypes verifies a full ChatRequestEnvelope
// with text/image/file/tool-invocation parts and v6 fields survives JSON round-trip
// and converts correctly to AI messages.
func TestIntegration_EnvelopeRoundTrip_WithAllPartTypes(t *testing.T) {
	original := ChatRequestEnvelope{
		ID:        "sess-rt-1",
		Trigger:   "submit-message",
		MessageID: "",
		Messages: []EnvelopeMessage{
			{
				ID:   "msg-user-1",
				Role: "user",
				Parts: []EnvelopePartUnion{
					{Type: EnvelopePartTypeText, Text: "What is 2+2?"},
					{Type: EnvelopePartTypeImage, URL: "https://example.com/img.png", MediaType: "image/png"},
				},
				Metadata: map[string]any{"clientTime": "08:00"},
			},
			{
				ID:   "msg-asst-1",
				Role: "assistant",
				Parts: []EnvelopePartUnion{
					{
						Type:       EnvelopePartTypeToolInvocation,
						ToolCallID: "tc-rt-1",
						ToolName:   "add",
						Input:      json.RawMessage(`{"a":2,"b":2}`),
						Output:     "4",
						State:      "result",
					},
					{Type: EnvelopePartTypeText, Text: "The answer is 4."},
				},
			},
		},
		Body:     map[string]any{"modelId": "openai:gpt-4o", "maxSteps": float64(3)},
		Metadata: map[string]any{"userId": "user-42"},
	}

	// Round-trip via JSON.
	raw, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded ChatRequestEnvelope
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Verify v6 fields survived.
	if decoded.Trigger != "submit-message" {
		t.Errorf("Trigger: got %q", decoded.Trigger)
	}
	if decoded.Messages[0].Metadata["clientTime"] != "08:00" {
		t.Errorf("per-message metadata: got %v", decoded.Messages[0].Metadata)
	}

	// Verify tool-invocation part decoded correctly.
	toolPart := decoded.Messages[1].Parts[0]
	if toolPart.Type != EnvelopePartTypeToolInvocation {
		t.Errorf("tool part type: got %q", toolPart.Type)
	}
	if toolPart.ToolCallID != "tc-rt-1" {
		t.Errorf("ToolCallID: got %q", toolPart.ToolCallID)
	}
	if toolPart.State != "result" {
		t.Errorf("State: got %q", toolPart.State)
	}

	// Convert user message parts to AI content parts.
	userParts := ToAIContentParts(decoded.Messages[0].Parts)
	if len(userParts) != 2 {
		t.Errorf("user message AI parts: got %d, want 2", len(userParts))
	}

	// Convert assistant message tool-invocation (state=result) -> 2 parts (call + result).
	assistantParts := ToAIContentParts(decoded.Messages[1].Parts)
	if len(assistantParts) != 3 { // tool-call + tool-result + text
		t.Errorf("assistant message AI parts: got %d, want 3 (tool-call + tool-result + text)", len(assistantParts))
	}
}

// TestIntegration_SendStartFalse_SendFinishFalse_ForMergePattern verifies that
// suppressing both lifecycle chunks leaves only step content, suitable for merging.
func TestIntegration_SendStartFalse_SendFinishFalse_ForMergePattern(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "merged content"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-merge", ToUIStreamOptions{
		SendReasoning: true,
		SendSources:   true,
		SendStart:     boolPtr(false),
		SendFinish:    boolPtr(false),
	})
	chunks := drainChunks(ch)

	if _, ok := findChunk(chunks, ChunkStart); ok {
		t.Error("merge pattern: must not emit start chunk")
	}
	if _, ok := findChunk(chunks, ChunkFinish); ok {
		t.Error("merge pattern: must not emit finish chunk")
	}
	if _, ok := findChunk(chunks, ChunkTextDelta); !ok {
		t.Error("merge pattern: expected text-delta to pass through")
	}
}
