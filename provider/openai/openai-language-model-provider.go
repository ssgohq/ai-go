package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

const defaultBaseURL = "https://api.openai.com/v1"

// LanguageModel implements ai.LanguageModel for the OpenAI Responses API.
type LanguageModel struct {
	modelID      string
	apiKey       string
	baseURL      string
	chunkTimeout time.Duration
	client       *http.Client
}

// Config holds options for constructing an OpenAI LanguageModel.
type Config struct {
	APIKey       string
	BaseURL      string        // optional; defaults to https://api.openai.com/v1
	Timeout      time.Duration // optional; defaults to 120s
	ChunkTimeout time.Duration // optional; per-chunk SSE read timeout (0 = disabled)
}

// NewLanguageModel creates an OpenAI-backed ai.LanguageModel using the Responses API.
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
		modelID:      modelID,
		apiKey:       cfg.APIKey,
		baseURL:      base,
		chunkTimeout: cfg.ChunkTimeout,
		client:       &http.Client{Timeout: timeout},
	}
}

// ModelID returns the OpenAI model identifier.
func (m *LanguageModel) ModelID() string { return m.modelID }

// Stream sends a streaming Responses API request and returns a channel of
// normalized ai.StreamEvents. Warnings from request encoding are emitted as
// the first event when non-empty.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	apiReq, warnings, err := encodeRequest(m.modelID, req, true)
	if err != nil {
		return nil, fmt.Errorf("openai: encode request: %w", err)
	}

	resp, err := m.doRequest(ctx, apiReq) //nolint:bodyclose // body closed by decodeResponsesSSEStream
	if err != nil {
		return nil, err
	}

	body := resp.Body
	if m.chunkTimeout > 0 {
		body = openaichat.NewTimeoutReader(resp.Body, m.chunkTimeout)
	}

	ch := make(chan ai.StreamEvent, 64)
	go func() {
		// Encoding warnings are merged onto the response.completed finish event
		// inside the decoder so callers see them in GenerateTextResult.Warnings.
		decodeResponsesSSEStream(ctx, body, ch, warnings...)
	}()
	return ch, nil
}

// GenerateText sends a non-streaming Responses API request and returns the
// aggregated result directly. Use ai.GenerateText for the full tool-loop path.
func (m *LanguageModel) GenerateText(ctx context.Context, req ai.LanguageModelRequest) (*ai.GenerateTextResult, error) {
	apiReq, warnings, err := encodeRequest(m.modelID, req, false)
	if err != nil {
		return nil, fmt.Errorf("openai: encode request: %w", err)
	}

	resp, err := m.doRequest(ctx, apiReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response body: %w", err)
	}

	return decodeResponsesNonStream(body, warnings)
}

// doRequest marshals body, sends POST /responses, and returns the HTTP response.
// Non-2xx responses are converted to errors with the body included.
func (m *LanguageModel) doRequest(ctx context.Context, apiReq responsesRequest) (*http.Response, error) {
	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		m.baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: build http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: http request: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		errBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("openai: unexpected status %d (failed to read body: %w)", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("openai: unexpected status %d: %s", resp.StatusCode, string(errBody))
	}
	return resp, nil
}
