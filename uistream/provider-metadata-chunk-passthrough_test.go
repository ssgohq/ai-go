package uistream

import (
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// TestWithProviderMetadata_NilReturnsUnchanged verifies nil providerMetadata leaves fields intact.
func TestWithProviderMetadata_NilReturnsUnchanged(t *testing.T) {
	f := map[string]any{"id": "x"}
	got := withProviderMetadata(f, nil)
	if _, hasKey := got["providerMetadata"]; hasKey {
		t.Error("expected no providerMetadata key when pm is nil")
	}
}

// TestWithProviderMetadata_NilFields_AllocatesMap verifies a nil fields map is allocated when pm non-nil.
func TestWithProviderMetadata_NilFields_AllocatesMap(t *testing.T) {
	pm := map[string]any{"openai": map[string]any{"logprobs": 0.5}}
	got := withProviderMetadata(nil, pm)
	if got == nil {
		t.Fatal("expected non-nil map when pm is non-nil")
	}
	if got["providerMetadata"] == nil {
		t.Error("expected providerMetadata key set")
	}
}

// TestChunkProducer_ProviderMetadata_TextDelta verifies text-delta chunks carry providerMetadata.
func TestChunkProducer_ProviderMetadata_TextDelta(t *testing.T) {
	pm := map[string]any{"gemini": map[string]any{"safetyRating": "safe"}}
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "hi", ProviderMetadata: pm},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-pm1", ToUIStreamOptions{SendReasoning: true, SendSources: true})
	chunks := drainChunks(ch)

	delta, ok := findChunk(chunks, ChunkTextDelta)
	if !ok {
		t.Fatal("expected text-delta chunk")
	}
	if delta.Fields["providerMetadata"] == nil {
		t.Error("expected providerMetadata on text-delta chunk when engine provides it")
	}
}

// TestChunkProducer_ProviderMetadata_AbsentWhenNil verifies providerMetadata is absent when engine does not set it.
func TestChunkProducer_ProviderMetadata_AbsentWhenNil(t *testing.T) {
	sr := newMockStreamEventer(
		engine.StepEvent{Type: engine.StepEventStepStart},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "hi"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	ch := ToUIMessageStream(sr, "msg-pm2", ToUIStreamOptions{SendReasoning: true, SendSources: true})
	chunks := drainChunks(ch)

	delta, ok := findChunk(chunks, ChunkTextDelta)
	if !ok {
		t.Fatal("expected text-delta chunk")
	}
	if _, hasKey := delta.Fields["providerMetadata"]; hasKey {
		t.Error("expected no providerMetadata key when engine does not set it")
	}
}

// TestWriter_WriteChunkWithProviderMetadata_InSSEOutput verifies providerMetadata appears in SSE.
func TestWriter_WriteChunkWithProviderMetadata_InSSEOutput(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteChunkWithProviderMetadata(
			ChunkTextDelta,
			map[string]any{"id": "text_1", "delta": "hello"},
			map[string]any{"openai": map[string]any{"logprobs": 0.9}},
		)
	})
	if !strings.Contains(output, `"providerMetadata"`) {
		t.Errorf("expected providerMetadata in SSE output\ngot: %s", output)
	}
	if !strings.Contains(output, `"delta":"hello"`) {
		t.Errorf("expected delta field in SSE output\ngot: %s", output)
	}
}

// TestWriter_WriteChunkWithProviderMetadata_NilMetaOmitted verifies nil providerMeta produces no key.
func TestWriter_WriteChunkWithProviderMetadata_NilMetaOmitted(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteChunkWithProviderMetadata(
			ChunkTextDelta,
			map[string]any{"id": "text_1", "delta": "hi"},
			nil,
		)
	})
	if strings.Contains(output, `"providerMetadata"`) {
		t.Errorf("expected no providerMetadata key when nil\ngot: %s", output)
	}
}
