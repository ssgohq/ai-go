package ai

import "context"

// ToolDefinition describes a callable function tool available to the model.
type ToolDefinition struct {
	// Name is the function name the model uses to invoke the tool.
	Name string
	// Description explains what the tool does; the model uses this for selection.
	Description string
	// InputSchema is a JSON Schema object describing the tool's input parameters.
	// Use the schema package to build this map, or construct it manually.
	InputSchema map[string]any
}

// ToolChoice controls which (if any) tool the model must call.
// Use the ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired constants or
// ToolChoiceSpecific to require a named tool.
type ToolChoice struct {
	// Type is one of "auto", "none", "required", or "tool".
	Type string
	// ToolName is set when Type == "tool" to name the required tool.
	ToolName string
}

var (
	// ToolChoiceAuto lets the model decide whether and which tool to call (default).
	ToolChoiceAuto = ToolChoice{Type: "auto"}
	// ToolChoiceNone prevents the model from calling any tool.
	ToolChoiceNone = ToolChoice{Type: "none"}
	// ToolChoiceRequired forces the model to call at least one tool.
	ToolChoiceRequired = ToolChoice{Type: "required"}
)

// ToolChoiceSpecific returns a ToolChoice that forces the model to call toolName.
func ToolChoiceSpecific(toolName string) ToolChoice {
	return ToolChoice{Type: "tool", ToolName: toolName}
}

// ToolResult holds the output of a single tool invocation.
type ToolResult struct {
	ID     string
	Name   string
	Args   string
	Output string
}

// ToolExecutor executes a named tool with JSON arguments and returns a result string.
type ToolExecutor interface {
	Execute(ctx context.Context, name, argsJSON string) (string, error)
}

// ToolSet is a named collection of tool definitions and an executor.
type ToolSet struct {
	Definitions []ToolDefinition
	Executor    ToolExecutor
}
