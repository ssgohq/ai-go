// Package schema provides helpers for building JSON Schema definitions
// used with ai.OutputSchema for structured output calls.
package schema

// Object builds a JSON Schema object definition with the given properties.
func Object(properties map[string]any, required ...string) map[string]any {
	s := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		s["required"] = required
	}
	return s
}

// Array builds a JSON Schema array definition with the given item schema.
func Array(items map[string]any) map[string]any {
	return map[string]any{
		"type":  "array",
		"items": items,
	}
}

// String returns a JSON Schema string property definition.
func String(description string) map[string]any {
	s := map[string]any{"type": "string"}
	if description != "" {
		s["description"] = description
	}
	return s
}

// Number returns a JSON Schema number property definition.
func Number(description string) map[string]any {
	s := map[string]any{"type": "number"}
	if description != "" {
		s["description"] = description
	}
	return s
}

// Integer returns a JSON Schema integer property definition.
func Integer(description string) map[string]any {
	s := map[string]any{"type": "integer"}
	if description != "" {
		s["description"] = description
	}
	return s
}

// Boolean returns a JSON Schema boolean property definition.
func Boolean(description string) map[string]any {
	s := map[string]any{"type": "boolean"}
	if description != "" {
		s["description"] = description
	}
	return s
}

// Enum returns a JSON Schema enum property definition.
func Enum(values ...string) map[string]any {
	vals := make([]any, len(values))
	for i, v := range values {
		vals[i] = v
	}
	return map[string]any{
		"type": "string",
		"enum": vals,
	}
}
