// Package openai_compatible provides a generic OpenAI-compatible language model
// for custom gateways, vendor endpoints, and local inference servers that implement
// the OpenAI chat completions API.
package openai_compatible

import "time"

// Config holds options for constructing a generic OpenAI-compatible LanguageModel.
type Config struct {
	// APIKey is used for Authorization: Bearer <key>.
	APIKey string
	// BaseURL is the API endpoint base, e.g. "https://api.openrouter.ai/api/v1".
	BaseURL string
	// Timeout is the HTTP client timeout. Defaults to 120s.
	Timeout time.Duration
	// Headers holds additional HTTP headers to include on every request.
	Headers map[string]string
	// ProviderName is used in error messages and metadata.
	// Defaults to "openaiCompatible".
	ProviderName string
	// SupportsStructuredOutput controls whether json_schema response_format is used.
	// When false (default), the provider falls back to json_object when structured
	// output is requested. Set true only if the backend supports json_schema mode.
	SupportsStructuredOutput bool
	// SupportsStreamUsage controls whether stream_options.include_usage is sent.
	// Defaults to false for maximum compatibility with providers that may not support it.
	SupportsStreamUsage bool
	// TransformRequest is an optional escape-hatch hook to mutate the raw chat request
	// before it is sent. Use sparingly — prefer capability flags for known behaviors.
	TransformRequest func(req map[string]any) map[string]any
	// ChunkTimeout is the per-chunk SSE read timeout. Zero means disabled.
	ChunkTimeout time.Duration
}
