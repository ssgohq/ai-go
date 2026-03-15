package ai_test

import (
	"context"
	"errors"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// stubEmbeddingModel is a test double for ai.EmbeddingModel.
type stubEmbeddingModel struct {
	modelID string
	// embedFn is called by Embed; if nil, returns a fixed vector.
	embedFn func(ctx context.Context, text string) ([]float32, error)
	// batchFn is called by EmbedBatch; if nil, delegates to embedFn per text.
	batchFn func(ctx context.Context, texts []string) ([][]float32, error)
}

func (s *stubEmbeddingModel) ModelID() string { return s.modelID }

func (s *stubEmbeddingModel) Embed(ctx context.Context, text string) ([]float32, error) {
	if s.embedFn != nil {
		return s.embedFn(ctx, text)
	}
	return []float32{0.1, 0.2, 0.3}, nil
}

func (s *stubEmbeddingModel) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if s.batchFn != nil {
		return s.batchFn(ctx, texts)
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{float32(i) * 0.1, float32(i) * 0.2}
	}
	return out, nil
}

func TestEmbed_Success(t *testing.T) {
	model := &stubEmbeddingModel{modelID: "test-model"}
	result, err := ai.Embed(context.Background(), ai.EmbedRequest{
		Model: model,
		Text:  "hello world",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ModelID != "test-model" {
		t.Errorf("expected ModelID=%q, got %q", "test-model", result.ModelID)
	}
	if len(result.Embedding) == 0 {
		t.Error("expected non-empty embedding")
	}
}

func TestEmbed_PropagatesError(t *testing.T) {
	want := errors.New("embed failed")
	model := &stubEmbeddingModel{
		modelID: "m",
		embedFn: func(_ context.Context, _ string) ([]float32, error) {
			return nil, want
		},
	}
	_, err := ai.Embed(context.Background(), ai.EmbedRequest{Model: model, Text: "x"})
	if !errors.Is(err, want) {
		t.Errorf("expected sentinel error, got %v", err)
	}
}

func TestEmbedMany_Success(t *testing.T) {
	texts := []string{"doc one", "doc two", "doc three"}
	model := &stubEmbeddingModel{modelID: "batch-model"}

	result, err := ai.EmbedMany(context.Background(), ai.EmbedManyRequest{
		Model: model,
		Texts: texts,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ModelID != "batch-model" {
		t.Errorf("expected ModelID=%q, got %q", "batch-model", result.ModelID)
	}
	if len(result.Embeddings) != len(texts) {
		t.Errorf("expected %d embeddings, got %d", len(texts), len(result.Embeddings))
	}
}

func TestEmbedMany_OrderPreserved(t *testing.T) {
	// Each embedding encodes the input index so we can verify ordering.
	batchFn := func(_ context.Context, texts []string) ([][]float32, error) {
		out := make([][]float32, len(texts))
		for i := range texts {
			out[i] = []float32{float32(i + 1)}
		}
		return out, nil
	}
	model := &stubEmbeddingModel{modelID: "m", batchFn: batchFn}

	texts := []string{"a", "b", "c"}
	result, err := ai.EmbedMany(context.Background(), ai.EmbedManyRequest{Model: model, Texts: texts})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, emb := range result.Embeddings {
		if emb[0] != float32(i+1) {
			t.Errorf("index %d: expected embedding value %v, got %v", i, float32(i+1), emb[0])
		}
	}
}

func TestEmbedMany_PropagatesError(t *testing.T) {
	want := errors.New("batch failed")
	model := &stubEmbeddingModel{
		modelID: "m",
		batchFn: func(_ context.Context, _ []string) ([][]float32, error) {
			return nil, want
		},
	}
	_, err := ai.EmbedMany(context.Background(), ai.EmbedManyRequest{
		Model: model,
		Texts: []string{"x", "y"},
	})
	if !errors.Is(err, want) {
		t.Errorf("expected sentinel error, got %v", err)
	}
}

// TestEmbedMany_SearchInputs simulates typical search and document embedding
// inputs (short query strings and longer document chunks).
func TestEmbedMany_SearchInputs(t *testing.T) {
	searchInputs := []string{
		"find all messages about project alpha",
		"what did we decide about the database schema",
	}
	docChunks := []string{
		"Chapter 1: Introduction to distributed systems...",
		"Chapter 2: Consensus algorithms and their trade-offs...",
		"Chapter 3: Replication strategies for high availability...",
	}
	all := append(searchInputs, docChunks...)

	model := &stubEmbeddingModel{modelID: "text-embedding-004"}
	result, err := ai.EmbedMany(context.Background(), ai.EmbedManyRequest{
		Model: model,
		Texts: all,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != len(all) {
		t.Errorf("expected %d embeddings, got %d", len(all), len(result.Embeddings))
	}
	for i, emb := range result.Embeddings {
		if len(emb) == 0 {
			t.Errorf("embedding[%d] is empty", i)
		}
	}
}
