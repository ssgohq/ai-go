package uistream

import (
	"testing"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// mockStreamEventer implements StreamEventer for testing without importing package ai.
type mockStreamEventer struct {
	ch      chan engine.StepEvent
	drained bool
}

func newMockStreamEventer(evs ...engine.StepEvent) *mockStreamEventer {
	ch := make(chan engine.StepEvent, len(evs))
	for _, e := range evs {
		ch <- e
	}
	close(ch)
	return &mockStreamEventer{ch: ch}
}

func (m *mockStreamEventer) Events() <-chan engine.StepEvent { return m.ch }
func (m *mockStreamEventer) DrainUnused()                    { m.drained = true }

// drainChunks reads all chunks from ch into a slice.
func drainChunks(ch <-chan Chunk) []Chunk {
	var out []Chunk
	for c := range ch {
		out = append(out, c)
	}
	return out
}

// findChunk returns the first chunk with the given type, or nil fields if not found.
func findChunk(chunks []Chunk, typ string) (Chunk, bool) {
	for _, c := range chunks {
		if c.Type == typ {
			return c, true
		}
	}
	return Chunk{}, false
}

// collectChunks returns all chunks with the given type.
func collectChunks(chunks []Chunk, typ string) []Chunk {
	var out []Chunk
	for _, c := range chunks {
		if c.Type == typ {
			out = append(out, c)
		}
	}
	return out
}

// TestToUIMessageStream_TextDeltas verifies text deltas become text-delta chunks.
func TestToUIMessageStream_TextDeltas(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "Hello "},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "world"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-1", ToUIStreamOptions{SendReasoning: true, SendSources: true})
	chunks := drainChunks(ch)

	if !sr.drained {
		t.Error("expected DrainUnused to be called")
	}

	deltas := collectChunks(chunks, ChunkTextDelta)
	if len(deltas) != 2 {
		t.Fatalf("expected 2 text-delta chunks, got %d", len(deltas))
	}
	if d, _ := deltas[0].Fields["delta"].(string); d != "Hello " {
		t.Errorf("expected first delta=%q, got %q", "Hello ", d)
	}
	if d, _ := deltas[1].Fields["delta"].(string); d != "world" {
		t.Errorf("expected second delta=%q, got %q", "world", d)
	}

	// Should have start and finish.
	if _, ok := findChunk(chunks, ChunkStart); !ok {
		t.Error("expected start chunk")
	}
	if _, ok := findChunk(chunks, ChunkFinish); !ok {
		t.Error("expected finish chunk")
	}
}

// TestToUIMessageStream_SendReasoningTrue forwards reasoning chunks.
func TestToUIMessageStream_SendReasoningTrue(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventReasoningDelta, ReasoningDelta: "thinking..."},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "answer"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-2", ToUIStreamOptions{SendReasoning: true, SendSources: true})
	chunks := drainChunks(ch)

	if _, ok := findChunk(chunks, ChunkReasoningStart); !ok {
		t.Error("expected reasoning-start chunk")
	}
	if _, ok := findChunk(chunks, ChunkReasoningDelta); !ok {
		t.Error("expected reasoning-delta chunk")
	}
	if _, ok := findChunk(chunks, ChunkReasoningEnd); !ok {
		t.Error("expected reasoning-end chunk")
	}
}

// TestToUIMessageStream_SendReasoningFalse suppresses reasoning events.
func TestToUIMessageStream_SendReasoningFalse(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventReasoningDelta, ReasoningDelta: "thinking..."},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "answer"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-3", ToUIStreamOptions{SendReasoning: false, SendSources: true})
	chunks := drainChunks(ch)

	if _, ok := findChunk(chunks, ChunkReasoningStart); ok {
		t.Error("expected no reasoning-start chunk when SendReasoning=false")
	}
	if _, ok := findChunk(chunks, ChunkReasoningDelta); ok {
		t.Error("expected no reasoning-delta chunk when SendReasoning=false")
	}
	// Text should still come through.
	if _, ok := findChunk(chunks, ChunkTextDelta); !ok {
		t.Error("expected text-delta chunk")
	}
}

// TestToUIMessageStream_SendSourcesTrue forwards source events.
func TestToUIMessageStream_SendSourcesTrue(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventSource, Source: &engine.Source{
			ID: "src-1", URL: "https://example.com", Title: "Example",
		}},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "text"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-4", ToUIStreamOptions{SendReasoning: true, SendSources: true})
	chunks := drainChunks(ch)

	if _, ok := findChunk(chunks, ChunkSourceURL); !ok {
		t.Error("expected source-url chunk when SendSources=true")
	}
}

// TestToUIMessageStream_SendSourcesFalse suppresses source events.
func TestToUIMessageStream_SendSourcesFalse(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventSource, Source: &engine.Source{
			ID: "src-1", URL: "https://example.com", Title: "Example",
		}},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "text"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-5", ToUIStreamOptions{SendReasoning: true, SendSources: false})
	chunks := drainChunks(ch)

	if _, ok := findChunk(chunks, ChunkSourceURL); ok {
		t.Error("expected no source-url chunk when SendSources=false")
	}
	if _, ok := findChunk(chunks, ChunkTextDelta); !ok {
		t.Error("expected text-delta chunk")
	}
}

// TestToUIMessageStream_MessageMetadata attaches metadata to finish chunk.
func TestToUIMessageStream_MessageMetadata(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "hi"},
		engine.StepEvent{Type: engine.StepEventUsage, Usage: &engine.Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
			ReasoningTokens:  3,
		}},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	var capturedInfo MessageMetadataInfo
	ch := ToUIMessageStream(sr, "msg-6", ToUIStreamOptions{
		SendReasoning: true,
		SendSources:   true,
		MessageMetadata: func(info MessageMetadataInfo) map[string]any {
			capturedInfo = info
			return map[string]any{"model": "gpt-4o", "tokens": info.Usage.TotalTokens}
		},
	})
	chunks := drainChunks(ch)

	finish, ok := findChunk(chunks, ChunkFinish)
	if !ok {
		t.Fatal("expected finish chunk")
	}

	md, ok := finish.Fields["messageMetadata"].(map[string]any)
	if !ok {
		t.Fatal("expected messageMetadata on finish chunk")
	}
	if md["model"] != "gpt-4o" {
		t.Errorf("expected model=gpt-4o, got %v", md["model"])
	}
	if md["tokens"] != 15 {
		t.Errorf("expected tokens=15, got %v", md["tokens"])
	}

	// Verify usage was tracked.
	if capturedInfo.FinishReason != "stop" {
		t.Errorf("expected FinishReason=stop, got %q", capturedInfo.FinishReason)
	}
	if capturedInfo.Usage.PromptTokens != 10 {
		t.Errorf("expected PromptTokens=10, got %d", capturedInfo.Usage.PromptTokens)
	}
	if capturedInfo.Usage.ReasoningTokens != 3 {
		t.Errorf("expected ReasoningTokens=3, got %d", capturedInfo.Usage.ReasoningTokens)
	}
}

// TestToUIMessageStream_ChannelCloses verifies the output channel closes when the stream completes.
func TestToUIMessageStream_ChannelCloses(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "x"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-7", ToUIStreamOptions{SendReasoning: true, SendSources: true})

	// Drain all — this should not hang.
	count := 0
	for range ch {
		count++
	}
	if count == 0 {
		t.Error("expected at least one chunk")
	}
}

// boolPtr returns a pointer to the given bool value.
func boolPtr(b bool) *bool { return &b }

// TestToUIMessageStream_SendStartFalse suppresses the start chunk.
func TestToUIMessageStream_SendStartFalse(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "hi"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-sf1", ToUIStreamOptions{
		SendReasoning: true,
		SendSources:   true,
		SendStart:     boolPtr(false),
	})
	chunks := drainChunks(ch)

	if _, ok := findChunk(chunks, ChunkStart); ok {
		t.Error("expected no start chunk when SendStart=false")
	}
	if _, ok := findChunk(chunks, ChunkFinish); !ok {
		t.Error("expected finish chunk to still be present")
	}
}

// TestToUIMessageStream_SendFinishFalse suppresses the finish chunk.
func TestToUIMessageStream_SendFinishFalse(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "hi"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-sf2", ToUIStreamOptions{
		SendReasoning: true,
		SendSources:   true,
		SendFinish:    boolPtr(false),
	})
	chunks := drainChunks(ch)

	if _, ok := findChunk(chunks, ChunkFinish); ok {
		t.Error("expected no finish chunk when SendFinish=false")
	}
	if _, ok := findChunk(chunks, ChunkStart); !ok {
		t.Error("expected start chunk to still be present")
	}
}

// TestToUIMessageStream_DefaultSendStartFinish verifies both are emitted by default.
func TestToUIMessageStream_DefaultSendStartFinish(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "hi"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-sf3", ToUIStreamOptions{
		SendReasoning: true,
		SendSources:   true,
		// SendStart and SendFinish are nil — default is true
	})
	chunks := drainChunks(ch)

	if _, ok := findChunk(chunks, ChunkStart); !ok {
		t.Error("expected start chunk with default options")
	}
	if _, ok := findChunk(chunks, ChunkFinish); !ok {
		t.Error("expected finish chunk with default options")
	}
}

// TestToUIMessageStream_NilMetadataCallback uses default path without metadata.
func TestToUIMessageStream_NilMetadataCallback(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "plain"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-8", ToUIStreamOptions{
		SendReasoning: true,
		SendSources:   true,
	})
	chunks := drainChunks(ch)

	finish, ok := findChunk(chunks, ChunkFinish)
	if !ok {
		t.Fatal("expected finish chunk")
	}
	if _, hasMetadata := finish.Fields["messageMetadata"]; hasMetadata {
		t.Error("expected no messageMetadata when callback is nil")
	}
}
