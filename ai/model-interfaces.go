package ai

import "context"

// LanguageModel is the interface a provider must implement for chat/text generation.
type LanguageModel interface {
	// ModelID returns the provider-specific model identifier.
	ModelID() string

	// Stream starts a streaming chat completion and returns a channel of StreamEvents.
	Stream(ctx context.Context, req LanguageModelRequest) (<-chan StreamEvent, error)
}

// EmbeddingModel is the interface a provider must implement for text embeddings.
type EmbeddingModel interface {
	// ModelID returns the provider-specific model identifier.
	ModelID() string

	// Embed generates an embedding vector for a single text.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch generates embedding vectors for multiple texts.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// LanguageModelRequest is the normalized input to a LanguageModel.
type LanguageModelRequest struct {
	System   string
	Messages []Message
	Tools    []ToolDefinition
	Output   *OutputSchema
	Settings CallSettings
}

// StreamEventType identifies the kind of event emitted during streaming.
type StreamEventType int

const (
	StreamEventTextDelta       StreamEventType = iota
	StreamEventReasoningDelta
	StreamEventToolCallDelta
	StreamEventUsage
	StreamEventFinish
	StreamEventError
)

// StreamEvent is a single normalized event from a LanguageModel stream.
type StreamEvent struct {
	Type StreamEventType

	// TextDelta is set for StreamEventTextDelta and StreamEventReasoningDelta.
	TextDelta string

	// Tool call fields are set for StreamEventToolCallDelta.
	ToolCallIndex    int
	ToolCallID       string
	ToolCallName     string
	ToolCallArgsDelta string
	ThoughtSignature string

	// Usage is set for StreamEventUsage.
	Usage *Usage

	// FinishReason is set for StreamEventFinish.
	FinishReason FinishReason

	// Error is set for StreamEventError.
	Error error
}
