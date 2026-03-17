package gemini

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

const nativeBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// NativeLanguageModel implements ai.LanguageModel using the native Gemini API
// (:streamGenerateContent endpoint). Unlike the OpenAI-compatible LanguageModel,
// this provider fully supports Google Search grounding, native thinking config,
// and other Gemini-only features that are unavailable via the OpenAI compatibility
// layer.
//
// Use NewNativeLanguageModel to construct an instance.
type NativeLanguageModel struct {
	modelID string
	cfg     Config
	client  *http.Client
}

// NewNativeLanguageModel creates a Gemini-backed ai.LanguageModel that uses the
// native Gemini API directly (not the OpenAI-compatible endpoint).
//
// Use this when you need features like Google Search grounding or native thinking
// configuration. For basic chat completions, NewLanguageModel (OpenAI-compatible)
// may also work.
func NewNativeLanguageModel(modelID string, cfg Config) *NativeLanguageModel {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	return &NativeLanguageModel{
		modelID: modelID,
		cfg:     cfg,
		client:  &http.Client{Timeout: timeout},
	}
}

// ModelID returns the Gemini model identifier.
func (m *NativeLanguageModel) ModelID() string { return m.modelID }

// Stream sends a streaming request to the native Gemini API and returns a
// channel of normalized ai.StreamEvents.
func (m *NativeLanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	// Build native request body.
	nr := encodeNativeRequest(req)

	// Encode tools + toolConfig.
	opts := parseProviderOptions(req.ProviderOptions)
	toolResult := encodeNativeTools(req.Tools, req.ToolChoice, opts)
	nr.Tools = toolResult.Tools
	nr.ToolConfig = toolResult.ToolConfig

	// Generate warnings for unsupported option combinations.
	warnings := warningsForRequest(m.modelID, req)

	body, err := json.Marshal(nr)
	if err != nil {
		return nil, fmt.Errorf("gemini-native: marshal request: %w", err)
	}

	baseURL := m.cfg.BaseURL
	if baseURL == "" {
		baseURL = nativeBaseURL
	}
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", baseURL, m.modelID)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini-native: build http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", m.cfg.APIKey)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini-native: http request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("gemini-native: unexpected status %d (failed to read body: %w)", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("gemini-native: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan ai.StreamEvent, 64)
	go func() {
		decodeNativeSSEStream(ctx, resp.Body, ch)
		// Inject warnings into a synthetic finish event if the decoder didn't
		// emit one that we could attach to. In practice, warnings are rare and
		// the stream normally ends with a finish event.
		// We handle it inline by wrapping the channel to inject warnings into the
		// first finish event.
	}()

	if len(warnings) == 0 {
		return ch, nil
	}

	// Wrap channel to inject warnings into the first finish event.
	out := make(chan ai.StreamEvent, 64)
	go func() {
		defer close(out)
		finishInjected := false
		for ev := range ch {
			if !finishInjected && ev.Type == ai.StreamEventFinish {
				ev.Warnings = append(warnings, ev.Warnings...)
				finishInjected = true
			}
			out <- ev
		}
	}()
	return out, nil
}
