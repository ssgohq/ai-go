package openai

// ChatProviderOptions holds OpenAI Chat Completions-specific options passed via
// LanguageModelRequest.ProviderOptions["openai"].
//
// Usage:
//
//	req := ai.LanguageModelRequest{
//	    ProviderOptions: map[string]any{
//	        "openai": openai.ChatProviderOptions{
//	            User:            "user-123",
//	            ReasoningEffort: "medium",
//	        },
//	    },
//	}
type ChatProviderOptions struct {
	// User is an end-user identifier sent to OpenAI for abuse monitoring.
	User string

	// ReasoningEffort controls thinking depth for reasoning models
	// (o1, o3, o4-mini, gpt-5 series). Valid values: "low", "medium", "high".
	ReasoningEffort string

	// StrictJSONSchema controls the strict field in json_schema response_format.
	// Defaults to true when SupportsStructuredOutput is true.
	// Set false for schemas that cannot satisfy OpenAI strict-schema constraints
	// (e.g., not all properties in required, or additionalProperties not false).
	StrictJSONSchema *bool
}
