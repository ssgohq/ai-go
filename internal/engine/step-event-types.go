// Package engine implements the multi-step tool-loop that drives GenerateText and StreamText.
// It defines its own model interface and event types to avoid import cycles with package ai.
package engine

import "encoding/json"

// StepEventType identifies the kind of event emitted by the engine during a run.
type StepEventType int

const (
	StepEventTextDelta StepEventType = iota
	StepEventReasoningDelta
	StepEventToolCallStart // first delta for a new tool call
	StepEventToolCallDelta // subsequent argument fragment
	StepEventToolCallReady // tool call complete, about to execute
	StepEventToolResult    // tool execution result
	StepEventUsage
	StepEventStepStart
	StepEventStepEnd
	StepEventStructuredOutput
	StepEventDone
	StepEventError
	// StepEventSource carries a source reference from a provider-native tool.
	StepEventSource
)

// StepEvent is a single event emitted by the engine's Run goroutine.
type StepEvent struct {
	Type StepEventType

	// Text/reasoning fields.
	TextDelta      string
	ReasoningDelta string

	// Tool call fields.
	ToolCallIndex     int
	ToolCallID        string
	ToolCallName      string
	ToolCallArgsDelta string
	ThoughtSignature  string

	// Tool result.
	ToolResult *ToolResult

	// Usage.
	Usage *Usage

	// Step metadata.
	StepNumber   int
	FinishReason FinishReason
	// RawFinishReason is the unmodified finish reason string from the provider.
	RawFinishReason string
	// ProviderMetadata carries provider-specific metadata.
	ProviderMetadata map[string]any
	// Warnings carries non-fatal advisories.
	Warnings []Warning

	// Source is set for StepEventSource events.
	Source *Source

	// Structured output (final step only).
	StructuredOutput json.RawMessage

	// Error.
	Error error
}

// FinishReason indicates why the model stopped generating.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonError         FinishReason = "error"
	FinishReasonUnknown       FinishReason = "unknown"
)

// Usage holds token counts for a completion step.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// ToolResult holds the output of a single tool invocation.
type ToolResult struct {
	ID     string
	Name   string
	Args   string
	Output string
}

// StreamEventType identifies provider stream event kinds.
type StreamEventType int

const (
	StreamEventTextDelta StreamEventType = iota
	StreamEventReasoningDelta
	StreamEventToolCallDelta
	StreamEventUsage
	StreamEventFinish
	StreamEventError
	// StreamEventSource carries a source reference (web search result, document, etc.)
	StreamEventSource
)

// Source represents a single source reference from a provider-native tool.
type Source struct {
	SourceType       string
	ID               string
	URL              string
	Title            string
	ProviderMetadata map[string]any
}

// Warning is a non-fatal advisory from a provider.
type Warning struct {
	Type    string
	Message string
	Setting string
}

// StreamEvent is a normalized event from a Model stream.
type StreamEvent struct {
	Type StreamEventType

	TextDelta         string
	ToolCallIndex     int
	ToolCallID        string
	ToolCallName      string
	ToolCallArgsDelta string
	ThoughtSignature  string
	Usage             *Usage
	FinishReason      FinishReason
	// RawFinishReason is the unmodified finish reason string from the provider.
	RawFinishReason string
	// ProviderMetadata carries provider-specific metadata.
	ProviderMetadata map[string]any
	// Warnings carries non-fatal advisories from the provider.
	Warnings []Warning
	// Source is set for StreamEventSource events.
	Source *Source
	Error  error
}
