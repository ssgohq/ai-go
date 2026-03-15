package openai

import (
	"context"

	"github.com/ssgohq/ai-go/ai"
	"github.com/ssgohq/ai-go/provider/internal/openaichat"
)

const chatProviderName = "openai-chat"

// ChatLanguageModel implements ai.LanguageModel for the OpenAI Chat Completions API.
// Use NewChatLanguageModel to construct one.
type ChatLanguageModel struct {
	inner *openaichat.LanguageModel
}

// NewChatLanguageModel creates an OpenAI-backed ai.LanguageModel using the
// Chat Completions API (/chat/completions).
//
// This is distinct from NewLanguageModel which targets the Responses API.
// Use Chat Completions when you need:
//   - Broad model compatibility (gpt-3.5-turbo, gpt-4o, gpt-4.1, o-series)
//   - OpenAI-standard request/response shape
//   - Structured output via response_format (gpt-4o-2024-08-06+)
//
// Use NewLanguageModel (Responses API) when you need:
//   - previous_response_id continuation
//   - Built-in web search or file inputs
//   - Responses-native multi-modal output
func NewChatLanguageModel(modelID string, cfg Config) *ChatLanguageModel {
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	inner := openaichat.NewLanguageModel(openaichat.ModelConfig{
		ModelID:      modelID,
		ProviderName: chatProviderName,
		BaseURL:      base,
		APIKey:       cfg.APIKey,
		Timeout:      cfg.Timeout,
		Capabilities: openaichat.CapabilityFlags{
			SupportsStructuredOutput: true,
			SupportsStreamUsage:      true,
		},
	})
	return &ChatLanguageModel{inner: inner}
}

// ModelID returns the OpenAI model identifier.
func (m *ChatLanguageModel) ModelID() string { return m.inner.ModelID() }

// Stream sends a streaming Chat Completions request and returns a channel of
// normalized ai.StreamEvents.
func (m *ChatLanguageModel) Stream(
	ctx context.Context,
	req ai.LanguageModelRequest,
) (<-chan ai.StreamEvent, error) {
	return m.inner.Stream(ctx, req)
}
