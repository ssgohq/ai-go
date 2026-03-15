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
	System          string
	Messages        []Message
	Tools           []ToolDefinition
	Output          *OutputSchema
	Settings        CallSettings
	ProviderOptions map[string]any // provider-specific options keyed by provider name
}

// Warning is a non-fatal advisory from the provider or SDK layer.
type Warning struct {
	// Type identifies the warning kind: "unsupported-setting", "other", etc.
	Type    string
	Message string
	// Setting is set when Type == "unsupported-setting".
	Setting string
}

// StreamEventType identifies the kind of event emitted during streaming.
type StreamEventType int

const (
	StreamEventTextDelta StreamEventType = iota
	StreamEventReasoningDelta
	StreamEventToolCallDelta
	StreamEventUsage
	StreamEventFinish
	StreamEventError
	// StreamEventSource carries a source (web search result, document reference, etc.)
	StreamEventSource
)

// Source represents a single source reference from a provider-native tool such as web search.
type Source struct {
	// SourceType is "url" for web results, "document" for file/doc references, or a provider-specific string.
	SourceType string
	// ID is an optional provider-assigned identifier for dedup.
	ID string
	// URL is set when SourceType == "url".
	URL string
	// Title is a human-readable title for the source.
	Title string
	// ProviderMetadata holds provider-specific extra fields.
	ProviderMetadata map[string]any
}

// StreamEvent is a single normalized event from a LanguageModel stream.
type StreamEvent struct {
	Type StreamEventType

	// TextDelta is set for StreamEventTextDelta and StreamEventReasoningDelta.
	TextDelta string

	// Tool call fields are set for StreamEventToolCallDelta.
	ToolCallIndex     int
	ToolCallID        string
	ToolCallName      string
	ToolCallArgsDelta string
	ThoughtSignature  string

	// Usage is set for StreamEventUsage.
	Usage *Usage

	// FinishReason is set for StreamEventFinish.
	FinishReason FinishReason
	// RawFinishReason is the unmodified finish reason string from the provider.
	RawFinishReason string
	// ProviderMetadata carries provider-specific metadata attached at finish.
	ProviderMetadata map[string]any
	// Warnings carries non-fatal advisories from the provider.
	Warnings []Warning

	// Source is set for StreamEventSource.
	Source *Source

	// Error is set for StreamEventError.
	Error error
}
