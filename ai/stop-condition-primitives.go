package ai

// StopCondition determines when the tool loop should stop after each step.
// step is 1-based (first completed step = 1).
type StopCondition func(step int, result *StepResult) bool

// StepResult holds information about a completed tool-loop step.
type StepResult struct {
	HasToolCalls bool
	ToolNames    []string
	Text         string
}

// StepCountIs returns a StopCondition that stops after n completed steps.
func StepCountIs(n int) StopCondition {
	return func(step int, _ *StepResult) bool {
		return step >= n
	}
}

// Never returns a StopCondition that never stops early (run until no tool calls or maxSteps).
func Never() StopCondition {
	return func(_ int, _ *StepResult) bool {
		return false
	}
}

// HasToolCall returns a StopCondition that stops after a step that called the named tool.
// Useful for single-tool agentic flows that should halt once a specific action is taken.
func HasToolCall(toolName string) StopCondition {
	return func(_ int, r *StepResult) bool {
		for _, name := range r.ToolNames {
			if name == toolName {
				return true
			}
		}
		return false
	}
}

// OutputSchema describes the desired output mode for a generation call.
//
// Supported Type values:
//   - "text"        — plain text; no JSON constraint (default when Output is nil)
//   - "json_object" — any valid JSON object; provider uses json_object response_format
//   - "object"      — JSON object conforming to Schema (uses json_schema response_format)
//   - "array"       — JSON array conforming to Schema (uses json_schema response_format)
type OutputSchema struct {
	// Type is one of "text", "json_object", "object", or "array".
	Type string
	// Schema is the JSON Schema definition. Required for "object" and "array"; nil otherwise.
	Schema map[string]any
}

// OutputText signals that plain text output is expected (no JSON constraint).
func OutputText() *OutputSchema {
	return &OutputSchema{Type: "text"}
}

// OutputJSONObject signals that any valid JSON object is expected.
// The provider uses json_object response_format (no schema validation).
// Use OutputObject for schema-constrained JSON.
func OutputJSONObject() *OutputSchema {
	return &OutputSchema{Type: "json_object"}
}

// OutputObject creates an OutputSchema for a JSON object conforming to schema.
func OutputObject(schema map[string]any) *OutputSchema {
	return &OutputSchema{Type: "object", Schema: schema}
}

// OutputArray creates an OutputSchema for a JSON array with the given item schema.
func OutputArray(itemSchema map[string]any) *OutputSchema {
	return &OutputSchema{
		Type: "array",
		Schema: map[string]any{
			"type":  "array",
			"items": itemSchema,
		},
	}
}
