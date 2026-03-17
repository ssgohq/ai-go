package gemini

// ProviderOptions holds Gemini-specific options passed via
// GenerateTextRequest.ProviderOptions["gemini"].
type ProviderOptions struct {
	// EnableGoogleSearch enables the Google Search grounding tool.
	EnableGoogleSearch bool

	// GoogleSearchConfig holds optional search configuration.
	GoogleSearchConfig *GoogleSearchConfig

	// ThinkingConfig controls the model's thinking/reasoning behavior.
	ThinkingConfig *ThinkingConfig
}

// ThinkingConfig controls how the model uses its thinking/reasoning capability.
// See: https://ai.google.dev/gemini-api/docs/gemini-3?thinking=high#thinking_level
type ThinkingConfig struct {
	// ThinkingBudget sets a token budget for thinking. Optional.
	ThinkingBudget *int `json:"thinkingBudget,omitempty"`
	// IncludeThoughts controls whether thinking tokens are included in the response.
	IncludeThoughts *bool `json:"includeThoughts,omitempty"`
	// ThinkingLevel sets a preset thinking level: "minimal", "low", "medium", "high".
	ThinkingLevel string `json:"thinkingLevel,omitempty"`
}

// GoogleSearchConfig contains optional configuration for Google Search grounding.
type GoogleSearchConfig struct {
	// DynamicRetrievalThreshold controls when grounding is triggered (0.0-1.0).
	// When set, MODE_DYNAMIC is used; otherwise the grounding always applies.
	DynamicRetrievalThreshold *float64

	// SearchTypes specifies which search types to use (e.g. "web", "image").
	SearchTypes []string

	// TimeRangeFilter restricts search results to a specific time range.
	TimeRangeFilter *TimeRangeFilter
}

// TimeRangeFilter restricts Google Search grounding results to a time range.
type TimeRangeFilter struct {
	// StartTime is the start of the time range in RFC3339 format.
	StartTime string
	// EndTime is the end of the time range in RFC3339 format.
	EndTime string
}

// parseProviderOptions extracts Gemini-specific options from a provider options map.
func parseProviderOptions(opts map[string]any) ProviderOptions {
	if opts == nil {
		return ProviderOptions{}
	}
	v, ok := opts["gemini"]
	if !ok {
		return ProviderOptions{}
	}
	if p, ok := v.(ProviderOptions); ok {
		return p
	}
	return ProviderOptions{}
}
