package openaichat

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/open-ai-sdk/ai-go/ai"
)

// ModelConfig holds all configuration for the shared chat completions LanguageModel.
type ModelConfig struct {
	// ModelID is the model identifier sent to the API.
	ModelID string
	// ProviderName is used in error messages and metadata (e.g. "gemini", "openai").
	ProviderName string
	// BaseURL is the API endpoint base (e.g. "https://api.openai.com/v1").
	BaseURL string
	// APIKey is used for Authorization: Bearer <key>.
	APIKey string
	// Headers holds additional HTTP headers to include on every request.
	Headers map[string]string
	// Timeout is the HTTP client timeout. Defaults to 120s.
	Timeout time.Duration
	// Capabilities declares optional feature support.
	Capabilities CapabilityFlags
	// SanitizeTools is an optional hook to clean tool schemas before sending.
	SanitizeTools func(tools []map[string]any) []map[string]any
	// TransformRequestBody is an optional hook to mutate the request body map before
	// sending. The map is a JSON-serializable representation of ChatRequest.
	// Extra keys added here are preserved in the outgoing request body.
	TransformRequestBody func(body map[string]any) map[string]any
	// MetadataExtractor is an optional hook to extract provider metadata from SSE chunks.
	MetadataExtractor func(chunk StreamChunk) map[string]any
}

// LanguageModel implements ai.LanguageModel using OpenAI-style chat completions.
type LanguageModel struct {
	cfg    ModelConfig
	client *http.Client
}

// NewLanguageModel creates a LanguageModel with the given configuration.
func NewLanguageModel(cfg ModelConfig) *LanguageModel {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	return &LanguageModel{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
	}
}

// ModelID returns the configured model identifier.
func (m *LanguageModel) ModelID() string { return m.cfg.ModelID }

// Stream sends a streaming chat request and returns a channel of normalized ai.StreamEvents.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	params := EncodeRequestParams{
		ModelID:            m.cfg.ModelID,
		SanitizeTools:      m.cfg.SanitizeTools,
		IncludeStreamUsage: m.cfg.Capabilities.SupportsStreamUsage,
	}

	cr, err := EncodeRequest(params, req, true)
	if err != nil {
		return nil, fmt.Errorf("%s: encode request: %w", m.cfg.ProviderName, err)
	}

	var body []byte
	if m.cfg.TransformRequestBody != nil {
		// Marshal to map, apply transform, then re-marshal so extra fields survive.
		rawMap, mapErr := structToMap(cr)
		if mapErr != nil {
			return nil, fmt.Errorf("%s: marshal request to map: %w", m.cfg.ProviderName, mapErr)
		}
		rawMap = m.cfg.TransformRequestBody(rawMap)
		body, err = json.Marshal(rawMap)
	} else {
		body, err = json.Marshal(cr)
	}
	if err != nil {
		return nil, fmt.Errorf("%s: marshal request: %w", m.cfg.ProviderName, err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		m.cfg.BaseURL+"/chat/completions",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("%s: build http request: %w", m.cfg.ProviderName, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.cfg.APIKey)
	for k, v := range m.cfg.Headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%s: http request: %w", m.cfg.ProviderName, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf(
				"%s: unexpected status %d (failed to read body: %w)",
				m.cfg.ProviderName, resp.StatusCode, readErr,
			)
		}
		return nil, fmt.Errorf(
			"%s: unexpected status %d: %s",
			m.cfg.ProviderName, resp.StatusCode, string(respBody),
		)
	}

	ch := make(chan ai.StreamEvent, 64)
	go DecodeSSEStream(ctx, resp.Body, ch, SSEDecodeParams{
		ProviderName:      m.cfg.ProviderName,
		MetadataExtractor: m.cfg.MetadataExtractor,
	})
	return ch, nil
}

// structToMap marshals v to JSON and unmarshals into a map[string]any.
func structToMap(v any) (map[string]any, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return m, nil
}
