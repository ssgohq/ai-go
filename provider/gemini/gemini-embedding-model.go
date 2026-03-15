package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const embedBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// EmbeddingModel implements ai.EmbeddingModel using the Gemini native embedding API.
type EmbeddingModel struct {
	modelID string
	apiKey  string
	client  *http.Client
}

// NewEmbeddingModel creates a Gemini-backed ai.EmbeddingModel.
// modelID should be e.g. "text-embedding-004".
func NewEmbeddingModel(modelID string, cfg Config) *EmbeddingModel {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}
	return &EmbeddingModel{
		modelID: modelID,
		apiKey:  cfg.APIKey,
		client:  &http.Client{Timeout: timeout},
	}
}

// ModelID returns the Gemini embedding model identifier.
func (m *EmbeddingModel) ModelID() string { return m.modelID }

// Embed generates an embedding vector for a single text.
func (m *EmbeddingModel) Embed(ctx context.Context, text string) ([]float32, error) {
	results, err := m.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// EmbedBatch generates embedding vectors for multiple texts using Gemini's batchEmbedContents API.
// Results are parallel to input texts.
func (m *EmbeddingModel) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody, err := buildBatchEmbedRequest(m.modelID, texts)
	if err != nil {
		return nil, fmt.Errorf("gemini embed: build request: %w", err)
	}

	url := fmt.Sprintf("%s/models/%s:batchEmbedContents?key=%s", embedBaseURL, m.modelID, m.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("gemini embed: build http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini embed: http request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("gemini embed: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini embed: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	return parseBatchEmbedResponse(body, len(texts))
}

// geminiEmbedRequest is a single content entry for batchEmbedContents.
type geminiEmbedRequest struct {
	Model   string             `json:"model"`
	Content geminiEmbedContent `json:"content"`
}

type geminiEmbedContent struct {
	Parts []geminiEmbedPart `json:"parts"`
}

type geminiEmbedPart struct {
	Text string `json:"text"`
}

type geminiBatchEmbedRequest struct {
	Requests []geminiEmbedRequest `json:"requests"`
}

type geminiBatchEmbedResponse struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
}

func buildBatchEmbedRequest(modelID string, texts []string) ([]byte, error) {
	reqs := make([]geminiEmbedRequest, len(texts))
	for i, t := range texts {
		reqs[i] = geminiEmbedRequest{
			Model: "models/" + modelID,
			Content: geminiEmbedContent{
				Parts: []geminiEmbedPart{{Text: t}},
			},
		}
	}
	return json.Marshal(geminiBatchEmbedRequest{Requests: reqs})
}

func parseBatchEmbedResponse(body []byte, expectedCount int) ([][]float32, error) {
	var resp geminiBatchEmbedResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("gemini embed: parse response: %w", err)
	}
	if len(resp.Embeddings) != expectedCount {
		return nil, fmt.Errorf("gemini embed: expected %d embeddings, got %d", expectedCount, len(resp.Embeddings))
	}
	out := make([][]float32, len(resp.Embeddings))
	for i, e := range resp.Embeddings {
		out[i] = e.Values
	}
	return out, nil
}
