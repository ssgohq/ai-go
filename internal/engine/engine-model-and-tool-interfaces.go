package engine

import "context"

type ctxKey int

const toolCallIDCtxKey ctxKey = iota

// ToolCallIDFromContext returns the tool call ID injected by the engine
// when executing a tool. Returns empty string if not in a tool execution.
func ToolCallIDFromContext(ctx context.Context) string {
	v, ok := ctx.Value(toolCallIDCtxKey).(string)
	if !ok {
		return ""
	}
	return v
}

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
	Name          string
	Description   string
	InputSchema   map[string]any
	ToModelOutput func(result string) string
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
// Type is one of: "text", "image_url", "file", "tool_call", "tool_result", "reasoning".
type ContentPart struct {
	Type string

	// text / reasoning
	Text string

	// image_url
	ImageURL string

	// file
	FileURL  string
	MimeType string

	// Shared multimodal fields (image_url and file parts).
	Data     []byte // Inline binary content.
	FileID   string // Provider-specific file identifier.
	Filename string // Original filename (file parts).

	// tool_call
	ToolCallID       string
	ToolCallName     string
	ToolCallArgs     string // JSON string
	ThoughtSignature string // Gemini thought signature for multi-turn

	// tool_result
	ToolResultID     string
	ToolResultName   string
	ToolResultOutput string
}

// ToolChoice controls which tool the model must call.
type ToolChoice struct {
	// Type is one of "auto", "none", "required", or "tool".
	Type string
	// ToolName is set when Type == "tool".
	ToolName string
}

// Request is the engine-internal model request.
type Request struct {
	System          string
	Messages        []Message
	Tools           []ToolDefinition
	ToolChoice      *ToolChoice
	Output          *OutputSchema
	Settings        CallSettings
	ProviderOptions map[string]any
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
	TopP          *float32
	TopK          *int
	Seed          *int
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

// PrepareStepContext provides information about the current step for the PrepareStep callback.
type PrepareStepContext struct {
	StepNumber int
	Steps      []StepResultInfo
}

// StepResultInfo holds information about a completed step for PrepareStep evaluation.
type StepResultInfo struct {
	StepNumber   int
	HasToolCalls bool
	ToolNames    []string
	Text         string
	FinishReason FinishReason
}

// PrepareStepResult holds per-step overrides returned by PrepareStep.
// Nil fields mean "no override" — use the base request value.
type PrepareStepResult struct {
	Model           Model
	ToolChoice      *ToolChoice
	ActiveTools     []string
	System          string
	ProviderOptions map[string]any
}

// PrepareStepFunc is called before each step to allow per-step configuration overrides.
type PrepareStepFunc func(ctx PrepareStepContext) *PrepareStepResult

// LifecycleCallbacks holds optional callbacks for observability during a run.
type LifecycleCallbacks struct {
	OnStepFinish func(event StepFinishEvent)
	OnFinish     func(event FinishEvent)
	OnChunk      func(event StepEvent)
	OnError      func(err error)
}

// StepFinishEvent holds data passed to OnStepFinish after each step.
type StepFinishEvent struct {
	StepNumber       int
	ToolCalls        []ToolCallInfo
	ToolResults      []ToolResult
	FinishReason     FinishReason
	Usage            *Usage
	ProviderMetadata map[string]any
	Warnings         []Warning
}

// FinishEvent holds data passed to OnFinish when the entire run completes.
type FinishEvent struct {
	Text             string
	Reasoning        string
	Steps            []StepResultInfo
	TotalUsage       Usage
	FinishReason     FinishReason
	ProviderMetadata map[string]any
}

// ToolCallInfo describes a tool call for lifecycle callbacks.
type ToolCallInfo struct {
	ID   string
	Name string
	Args string
}

// RunParams configures a single engine run.
type RunParams struct {
	Model       Model
	Request     Request
	Tools       *ToolSet
	StopWhen    StopCondition
	MaxSteps    int
	PrepareStep PrepareStepFunc
	Callbacks   *LifecycleCallbacks
}
