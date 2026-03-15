package engine

import "context"

// Model is the engine-internal interface a language model provider must satisfy.
// It mirrors ai.LanguageModel but uses engine-local types to avoid import cycles.
type Model interface {
	ModelID() string
	Stream(ctx context.Context, req Request) (<-chan StreamEvent, error)
}

// ToolExecutor executes a named tool with JSON arguments.
type ToolExecutor interface {
	Execute(ctx context.Context, name, argsJSON string) (string, error)
}

// ToolDefinition describes a tool available to the model.
type ToolDefinition struct {
	Name        string
	Description string
	Parameters  map[string]any
}

// ToolSet is a collection of tool definitions and an executor.
type ToolSet struct {
	Definitions []ToolDefinition
	Executor    ToolExecutor
}

// Message is a conversation turn (engine-internal).
type Message struct {
	Role    string
	Content []ContentPart
}

// ContentPart is a single part of a message.
type ContentPart struct {
	Type string // "text", "image_url", "tool_call", "tool_result"

	// text
	Text string

	// image_url
	ImageURL string

	// tool_call
	ToolCallID   string
	ToolCallName string
	ToolCallArgs string // JSON string

	// tool_result
	ToolResultID     string
	ToolResultOutput string
}

// Request is the engine-internal model request.
type Request struct {
	System   string
	Messages []Message
	Tools    []ToolDefinition
	Output   *OutputSchema
	Settings CallSettings
}

// OutputSchema describes the desired JSON structure for a structured output call.
type OutputSchema struct {
	Type   string
	Schema map[string]any
}

// CallSettings controls model behavior per-request.
type CallSettings struct {
	Temperature   *float32
	MaxTokens     int
	StopSequences []string
}

// StopCondition determines when the tool loop should stop.
type StopCondition func(step int, result *StepResult) bool

// StepResult holds information about a completed step for stop condition evaluation.
type StepResult struct {
	HasToolCalls bool
	ToolNames    []string
	Text         string
}

// RunParams configures a single engine run.
type RunParams struct {
	Model    Model
	Request  Request
	Tools    *ToolSet
	StopWhen StopCondition
	MaxSteps int
}
