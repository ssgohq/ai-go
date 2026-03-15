package ai

import "context"

// ToolDefinition describes a tool available to the model.
type ToolDefinition struct {
	Name        string
	Description string
	Parameters  map[string]any // JSON Schema object
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
