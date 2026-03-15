package ai

import "context"

// EmbedRequest is the input for embedding a single text.
type EmbedRequest struct {
	Model EmbeddingModel
	Text  string
}

// EmbedManyRequest is the input for embedding multiple texts in batch.
type EmbedManyRequest struct {
	Model EmbeddingModel
	Texts []string
}

// EmbedResult holds the result of a single text embedding.
type EmbedResult struct {
	Embedding []float32
	// ModelID is the model that produced the embedding.
	ModelID string
}

// EmbedManyResult holds the result of a batch text embedding.
type EmbedManyResult struct {
	// Embeddings is parallel to the input texts slice.
	Embeddings [][]float32
	// ModelID is the model that produced the embeddings.
	ModelID string
}

// Embed generates an embedding vector for a single text.
func Embed(ctx context.Context, req EmbedRequest) (EmbedResult, error) {
	vec, err := req.Model.Embed(ctx, req.Text)
	if err != nil {
		return EmbedResult{}, err
	}
	return EmbedResult{
		Embedding: vec,
		ModelID:   req.Model.ModelID(),
	}, nil
}

// EmbedMany generates embedding vectors for multiple texts in a single batch call.
// Results are returned in the same order as the input texts.
func EmbedMany(ctx context.Context, req EmbedManyRequest) (EmbedManyResult, error) {
	vecs, err := req.Model.EmbedBatch(ctx, req.Texts)
	if err != nil {
		return EmbedManyResult{}, err
	}
	return EmbedManyResult{
		Embeddings: vecs,
		ModelID:    req.Model.ModelID(),
	}, nil
}
