package ai

import "encoding/json"

// GenerateTextRequest is the input to GenerateText and StreamText.
type GenerateTextRequest struct {
	// Model is the language model to call.
	Model LanguageModel
	// System is an optional system prompt prepended before the conversation.
	System string
	// Messages is the conversation history.
	Messages []Message
	// Tools is an optional set of callable functions for multi-step tool loops.
	Tools *ToolSet
	// ToolChoice controls which tool(s) the model may call. Defaults to ToolChoiceAuto.
	// Ignored when Tools is nil.
	ToolChoice *ToolChoice
	// StopWhen is an optional custom stop condition for the tool loop.
	StopWhen StopCondition
	// Output optionally constrains the model's output to a JSON schema or mode.
	Output *OutputSchema
	// Settings controls per-request model parameters (temperature, maxTokens, etc.).
	Settings CallSettings
	// MaxSteps limits the number of tool-loop iterations. Defaults to 10.
	MaxSteps int
	// ProviderOptions carries provider-specific options keyed by provider name.
	// Example: map[string]any{"openai": map[string]any{"previousResponseId": "r_abc"}}.
	ProviderOptions map[string]any
}

// StepOutput holds the result of a single tool-loop step.
type StepOutput struct {
	Text             string
	Reasoning        string
	ToolCalls        []ToolCallOutput
	ToolResults      []ToolResult
	Usage            Usage
	FinishReason     FinishReason
	RawFinishReason  string
	ProviderMetadata map[string]any
	Warnings         []Warning
	Sources          []Source
}

// ToolCallOutput holds the details of a single tool call made by the model.
type ToolCallOutput struct {
	ID   string
	Name string
	Args json.RawMessage
}

// GenerateTextResult holds the full output of a GenerateText call.
type GenerateTextResult struct {
	Text             string
	Reasoning        string
	Steps            []StepOutput
	ToolResults      []ToolResult
	TotalUsage       Usage
	FinishReason     FinishReason
	RawFinishReason  string
	ProviderMetadata map[string]any
	Warnings         []Warning
	Sources          []Source
	StructuredOutput json.RawMessage
}
