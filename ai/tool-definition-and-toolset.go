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
	// ToModelOutput optionally transforms the tool execution result before it
	// enters the conversation history. The original output is still reported in
	// ToolResult events. If nil, the raw output is used as-is.
	ToModelOutput func(result string) string
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

// ToolResultContent represents a single content part in a tool result.
type ToolResultContent struct {
	Type     string // "text" or "image"
	Text     string // for type="text"
	Data     []byte // for type="image"
	MimeType string // for type="image"
}

// ToolResult holds the output of a single tool invocation.
type ToolResult struct {
	ID      string
	Name    string
	Args    string
	Output  string
	Content []ToolResultContent // optional multi-part content
}

// ToolExecutor executes a named tool with JSON arguments and returns a result string.
type ToolExecutor interface {
	Execute(ctx context.Context, name, argsJSON string) (string, error)
}

// ToolResultStream allows tools to stream partial output to the UI in real-time.
type ToolResultStream interface {
	// Write sends a partial result to the UI (e.g., stdout from a bash command).
	Write(partial string)
}

// StreamingToolExecutor extends ToolExecutor with streaming support.
// Tools that implement this interface receive a stream for real-time output.
type StreamingToolExecutor interface {
	ToolExecutor
	// ExecuteStreaming executes a tool with a stream for partial results.
	// Falls back to Execute if not implemented.
	ExecuteStreaming(ctx context.Context, name, argsJSON string, stream ToolResultStream) (string, error)
}

// ToolSet is a named collection of tool definitions and an executor.
type ToolSet struct {
	Definitions []ToolDefinition
	Executor    ToolExecutor
}
