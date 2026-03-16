package gemini

import (
	"context"
	"time"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta/openai"

// LanguageModel implements ai.LanguageModel for the Gemini OpenAI-compatible API.
type LanguageModel struct {
	core *openaichat.LanguageModel
}

// Config holds options for constructing a Gemini LanguageModel or EmbeddingModel.
type Config struct {
	APIKey               string
	BaseURL              string        // optional; defaults to Gemini production endpoint
	Timeout              time.Duration // optional; defaults to 120s for LLM, 60s for embedding
	OutputDimensionality int           // optional; embedding output dimensions (768, 1536, 3072)
}

// NewLanguageModel creates a Gemini-backed ai.LanguageModel.
func NewLanguageModel(modelID string, cfg Config) *LanguageModel {
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	core := openaichat.NewLanguageModel(openaichat.ModelConfig{
		ModelID:      modelID,
		ProviderName: "gemini",
		BaseURL:      base,
		APIKey:       cfg.APIKey,
		Timeout:      cfg.Timeout,
		Capabilities: openaichat.CapabilityFlags{
			SupportsStructuredOutput: true,
			SupportsStreamUsage:      true,
		},
		SanitizeTools:        sanitizeToolSchemas,
		ExtraToolsForRequest: extraToolsForRequest,
	})
	return &LanguageModel{core: core}
}

// ModelID returns the Gemini model identifier.
func (m *LanguageModel) ModelID() string { return m.core.ModelID() }

// Stream sends a streaming chat request and returns a channel of normalized ai.StreamEvents.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	return m.core.Stream(ctx, req)
}

// extraToolsForRequest returns Gemini-specific built-in tools based on provider options.
func extraToolsForRequest(req ai.LanguageModelRequest) []map[string]any {
	opts := parseProviderOptions(req.ProviderOptions)
	if !opts.EnableGoogleSearch {
		return nil
	}
	return []map[string]any{{"type": "google_search"}}
}
