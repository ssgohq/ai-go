package openai_compatible

import (
	"context"

	"github.com/ssgohq/ai-go/ai"
	"github.com/ssgohq/ai-go/provider/internal/openaichat"
)

const defaultProviderName = "openaiCompatible"

// LanguageModel implements ai.LanguageModel for any OpenAI-compatible endpoint.
type LanguageModel struct {
	core *openaichat.LanguageModel
}

// NewLanguageModel creates an OpenAI-compatible ai.LanguageModel.
// The model delegates all chat-completions encoding and decoding to the
// shared openaichat core with the configuration provided.
func NewLanguageModel(modelID string, cfg Config) *LanguageModel {
	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = defaultProviderName
	}

	core := openaichat.NewLanguageModel(openaichat.ModelConfig{
		ModelID:      modelID,
		ProviderName: providerName,
		BaseURL:      cfg.BaseURL,
		APIKey:       cfg.APIKey,
		Timeout:      cfg.Timeout,
		Headers:      cfg.Headers,
		Capabilities: openaichat.CapabilityFlags{
			SupportsStructuredOutput: cfg.SupportsStructuredOutput,
			SupportsStreamUsage:      cfg.SupportsStreamUsage,
		},
		TransformRequestBody: cfg.TransformRequest,
	})

	return &LanguageModel{core: core}
}

// ModelID returns the configured model identifier.
func (m *LanguageModel) ModelID() string { return m.core.ModelID() }

// Stream sends a streaming chat request and returns a channel of normalized ai.StreamEvents.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	return m.core.Stream(ctx, req)
}
