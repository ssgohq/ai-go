// Package openai provides an ai-go LanguageModel implementation backed by the
// OpenAI Responses API, with support for previous-response continuation,
// reasoning settings, built-in web search, file-id inputs, and source inclusion.
package openai

// ProviderOptions holds OpenAI-specific options passed via
// GenerateTextRequest.ProviderOptions["openai"].
//
// Usage:
//
//	req := ai.GenerateTextRequest{
//	    Model: model,
//	    ProviderOptions: map[string]any{
//	        "openai": openai.ProviderOptions{
//	            PreviousResponseID: "resp_abc",
//	            ReasoningEffort:    "medium",
//	        },
//	    },
//	}
type ProviderOptions struct {
	// PreviousResponseID continues a prior Responses API response by id.
	// When set, the new request appends to that conversation thread.
	PreviousResponseID string

	// ReasoningEffort controls thinking depth for reasoning models
	// (o1, o3, o4-mini, gpt-5 series). Valid values: "low", "medium", "high".
	ReasoningEffort string

	// ReasoningSummary controls the format of reasoning summaries.
	// Valid values: "auto", "concise", "detailed".
	ReasoningSummary string

	// EnableWebSearch enables OpenAI's built-in web_search_preview tool.
	EnableWebSearch bool

	// IncludeSources requests that web search sources be included in the
	// response metadata. Only meaningful when EnableWebSearch is true.
	IncludeSources bool

	// MaxOutputTokens overrides CallSettings.MaxTokens for the Responses API.
	// If zero, CallSettings.MaxTokens is used.
	MaxOutputTokens int

	// Store controls whether the response is stored server-side.
	// Defaults to true (OpenAI default). Set to false to opt out.
	Store *bool

	// User is an end-user identifier for abuse monitoring.
	User string

	// Metadata is arbitrary key-value metadata stored with the generation.
	Metadata map[string]string
}

// parseProviderOptions extracts ProviderOptions from a generic provider options map.
// Returns zero-value ProviderOptions if the "openai" key is missing or wrong type.
func parseProviderOptions(opts map[string]any) ProviderOptions {
	if opts == nil {
		return ProviderOptions{}
	}
	v, ok := opts["openai"]
	if !ok {
		return ProviderOptions{}
	}
	switch p := v.(type) {
	case ProviderOptions:
		return p
	case map[string]any:
		return providerOptionsFromMap(p)
	}
	return ProviderOptions{}
}

func providerOptionsFromMap(m map[string]any) ProviderOptions {
	var p ProviderOptions
	if s, ok := m["previousResponseId"].(string); ok {
		p.PreviousResponseID = s
	}
	if s, ok := m["reasoningEffort"].(string); ok {
		p.ReasoningEffort = s
	}
	if s, ok := m["reasoningSummary"].(string); ok {
		p.ReasoningSummary = s
	}
	if b, ok := m["enableWebSearch"].(bool); ok {
		p.EnableWebSearch = b
	}
	if b, ok := m["includeSources"].(bool); ok {
		p.IncludeSources = b
	}
	if n, ok := m["maxOutputTokens"].(int); ok {
		p.MaxOutputTokens = n
	}
	if b, ok := m["store"].(*bool); ok {
		p.Store = b
	}
	if s, ok := m["user"].(string); ok {
		p.User = s
	}
	return p
}
