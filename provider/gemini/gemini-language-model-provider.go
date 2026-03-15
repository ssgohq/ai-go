package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/ssgohq/ai-go/ai"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta/openai"

// LanguageModel implements ai.LanguageModel for the Gemini OpenAI-compatible API.
type LanguageModel struct {
	modelID string
	apiKey  string
	baseURL string
	client  *http.Client
}

// Config holds options for constructing a Gemini LanguageModel.
type Config struct {
	APIKey  string
	BaseURL string // optional; defaults to Gemini production endpoint
	Timeout time.Duration
}

// NewLanguageModel creates a Gemini-backed ai.LanguageModel.
func NewLanguageModel(modelID string, cfg Config) *LanguageModel {
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	return &LanguageModel{
		modelID: modelID,
		apiKey:  cfg.APIKey,
		baseURL: base,
		client:  &http.Client{Timeout: timeout},
	}
}

// ModelID returns the Gemini model identifier.
func (m *LanguageModel) ModelID() string { return m.modelID }

// Stream sends a streaming chat request and returns a channel of normalized ai.StreamEvents.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	cr, err := encodeRequest(m.modelID, req, true)
	if err != nil {
		return nil, fmt.Errorf("gemini: encode request: %w", err)
	}

	body, err := json.Marshal(cr)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		m.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: build http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: http request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("gemini: unexpected status %d (failed to read body: %w)", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("gemini: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan ai.StreamEvent, 64)
	go decodeSSEStream(ctx, resp.Body, ch)
	return ch, nil
}
