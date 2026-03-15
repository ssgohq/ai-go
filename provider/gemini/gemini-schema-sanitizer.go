// Package gemini implements the ai.LanguageModel interface for the Gemini OpenAI-compatible API.
package gemini

// disallowedSchemaKeys are JSON Schema keys rejected by the Gemini API.
var disallowedSchemaKeys = map[string]bool{
	"$ref":                 true,
	"$defs":                true,
	"additionalProperties": true,
	"examples":             true,
	"default":              true,
}

// sanitizeToolSchemas returns deep copies of tool defs with disallowed keys removed.
func sanitizeToolSchemas(tools []map[string]any) []map[string]any {
	if tools == nil {
		return nil
	}
	out := make([]map[string]any, len(tools))
	for i, t := range tools {
		out[i] = sanitizeMap(t)
	}
	return out
}

func sanitizeMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if disallowedSchemaKeys[k] {
			continue
		}
		out[k] = sanitizeValue(v)
	}
	return out
}

func sanitizeValue(v any) any {
	switch val := v.(type) {
	case map[string]any:
		return sanitizeMap(val)
	case []any:
		return sanitizeSlice(val)
	default:
		return v
	}
}

func sanitizeSlice(s []any) []any {
	out := make([]any, len(s))
	for i, v := range s {
		out[i] = sanitizeValue(v)
	}
	return out
}
