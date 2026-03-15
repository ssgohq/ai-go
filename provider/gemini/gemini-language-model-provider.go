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
		SanitizeTools: sanitizeToolSchemas,
	})
	return &LanguageModel{core: core}
}

// ModelID returns the Gemini model identifier.
func (m *LanguageModel) ModelID() string { return m.core.ModelID() }

// Stream sends a streaming chat request and returns a channel of normalized ai.StreamEvents.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	return m.core.Stream(ctx, req)
}
