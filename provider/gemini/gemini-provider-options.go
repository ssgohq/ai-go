package gemini

// ProviderOptions holds Gemini-specific options passed via
// GenerateTextRequest.ProviderOptions["gemini"].
type ProviderOptions struct {
	// EnableGoogleSearch enables the Google Search grounding tool.
	EnableGoogleSearch bool

	// GoogleSearchConfig holds optional search configuration.
	GoogleSearchConfig *GoogleSearchConfig
}

// GoogleSearchConfig contains optional configuration for Google Search grounding.
type GoogleSearchConfig struct {
	// DynamicRetrievalThreshold is the threshold for dynamic retrieval mode (0.0-1.0).
	DynamicRetrievalThreshold *float64
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
