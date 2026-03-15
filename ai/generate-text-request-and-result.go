package ai

import "encoding/json"

// GenerateTextRequest is the input to GenerateText and StreamText.
type GenerateTextRequest struct {
	Model           LanguageModel
	System          string
	Messages        []Message
	Tools           *ToolSet
	StopWhen        StopCondition
	Output          *OutputSchema
	Settings        CallSettings
	MaxSteps        int
	ProviderOptions map[string]any // provider-specific options keyed by provider name
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
