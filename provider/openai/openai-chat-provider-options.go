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

// parseChatProviderOptions extracts ChatProviderOptions from a generic provider options map.
// Returns zero-value ChatProviderOptions if the "openai" key is missing or wrong type.
func parseChatProviderOptions(opts map[string]any) ChatProviderOptions {
	if opts == nil {
		return ChatProviderOptions{}
	}
	v, ok := opts["openai"]
	if !ok {
		return ChatProviderOptions{}
	}
	switch p := v.(type) {
	case ChatProviderOptions:
		return p
	case ProviderOptions:
		// Graceful fallback: allow callers using the Responses ProviderOptions type.
		return ChatProviderOptions{
			User:            p.User,
			ReasoningEffort: p.ReasoningEffort,
		}
	case map[string]any:
		return chatProviderOptionsFromMap(p)
	}
	return ChatProviderOptions{}
}

func chatProviderOptionsFromMap(m map[string]any) ChatProviderOptions {
	var p ChatProviderOptions
	if s, ok := m["user"].(string); ok {
		p.User = s
	}
	if s, ok := m["reasoningEffort"].(string); ok {
		p.ReasoningEffort = s
	}
	if b, ok := m["strictJSONSchema"].(*bool); ok {
		p.StrictJSONSchema = b
	}
	return p
}
