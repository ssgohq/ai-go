// Package anthropic implements a native ai.LanguageModel for the Anthropic Messages API.
// Unlike routing Anthropic through the OpenAI-compatible shim, this provider supports
// native features: prompt caching, extended thinking, and proper token counting.
package anthropic

import (
	"time"

	"github.com/open-ai-sdk/ai-go/ai"
)

// Config configures the Anthropic provider.
type Config struct {
	// APIKey is the Anthropic API key.
	APIKey string
	// BaseURL overrides the API endpoint. Default: "https://api.anthropic.com".
	BaseURL string
	// APIVersion is the Anthropic-Version header. Default: "2023-06-01".
	APIVersion string
	// Timeout is the HTTP client timeout. Default: 120s.
	Timeout time.Duration
	// EnableCaching enables prompt caching via cache_control headers.
	EnableCaching bool
}

func (c Config) withDefaults() Config {
	if c.BaseURL == "" {
		c.BaseURL = "https://api.anthropic.com"
	}
	if c.APIVersion == "" {
		c.APIVersion = "2023-06-01"
	}
	if c.Timeout == 0 {
		c.Timeout = 120 * time.Second
	}
	return c
}

// Provider creates Anthropic language models.
type Provider struct {
	config Config
}

// NewProvider creates a new Anthropic provider.
func NewProvider(cfg Config) *Provider {
	return &Provider{config: cfg.withDefaults()}
}

// LanguageModel returns an ai.LanguageModel for the given model ID.
// Supported: claude-4-sonnet, claude-4-opus, claude-3.7-sonnet, claude-3.5-sonnet, etc.
func (p *Provider) LanguageModel(modelID string) ai.LanguageModel {
	return NewLanguageModel(modelID, p.config)
}
