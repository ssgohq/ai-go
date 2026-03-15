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

// OutputSchema describes the desired JSON structure for a final structured-output call.
type OutputSchema struct {
	// Type is "object", "array", or "text".
	Type string
	// Schema is the JSON Schema definition (nil for "text").
	Schema map[string]any
}

// OutputObject creates an OutputSchema for a JSON object.
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

// OutputText signals that plain text output is expected (no schema constraint).
func OutputText() *OutputSchema {
	return &OutputSchema{Type: "text"}
}
