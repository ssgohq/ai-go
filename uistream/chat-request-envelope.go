package uistream

import "encoding/json"

// ChatRequestEnvelope is the canonical HTTP request body for the /chat endpoint.
// Both Go backends and Swift clients use this shape.
//
// The envelope carries enough context for:
//   - session identity (ID)
//   - conversation history (Messages)
//   - app-specific routing hints (Body)
//   - app-specific observability/tracking (Metadata)
//
// Backend route handlers decode this envelope and forward messages + body hints
// to the generation layer. Metadata is not forwarded to the model; it is used
// for logging, tracing, and persistence.
type ChatRequestEnvelope struct {
	// ID is the session or thread identifier. Used for conversation continuity.
	// Backends use this to load/persist conversation history.
	ID string `json:"id"`

	// Messages is the full conversation history to send to the model.
	// Each message has a role and a list of content parts (text, image, file, tool).
	Messages []EnvelopeMessage `json:"messages"`

	// Body carries route-level and model-level hints from the client.
	// Examples: modelId, agentId, runId, attachments, maxSteps.
	// The backend is free to read and discard keys it does not recognize.
	Body map[string]any `json:"body,omitempty"`

	// Metadata carries app-level context for logging and observability.
	// Examples: threadId, userId, experimentId.
	// This bag is NOT forwarded to the model.
	Metadata map[string]any `json:"metadata,omitempty"`

	// Trigger indicates the action that initiated this request.
	// Values: "submit-message" (default/new), "regenerate-message".
	// Backends can use this to skip persistence on regeneration.
	Trigger string `json:"trigger,omitempty"`

	// MessageID is the target message ID for regeneration.
	// Set when Trigger == "regenerate-message".
	// ResolveMessageID returns this when non-empty.
	MessageID string `json:"messageId,omitempty"`
}

// EnvelopeMessage is a single message inside ChatRequestEnvelope.
// Role is one of: "user", "assistant", "system".
// Parts carry the actual content; for simple text messages a single TextEnvelopePart suffices.
type EnvelopeMessage struct {
	// ID is the message identifier, used for continuation (resuming an existing assistant message).
	ID    string              `json:"id,omitempty"`
	Role  string              `json:"role"`
	Parts []EnvelopePartUnion `json:"parts,omitempty"`

	// Content is the flat string shorthand for a single-text-part message.
	// If Parts is non-empty, Content is ignored.
	Content string `json:"content,omitempty"`

	// Metadata carries per-message metadata for persistence and observability.
	// This is NOT forwarded to the model; it mirrors AI SDK Node v5's UIMessage.metadata.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ResolveMessageID returns the ID of the last assistant message for continuation,
// or the fallback ID if no continuation is applicable.
// This supports message continuation when the last message is from the assistant.
func ResolveMessageID(messages []EnvelopeMessage, fallback string) string {
	if len(messages) == 0 {
		return fallback
	}
	last := messages[len(messages)-1]
	if last.Role == "assistant" && last.ID != "" {
		return last.ID
	}
	return fallback
}

// ResolveMessageIDFromEnvelope resolves the message ID for a stream response.
// It prefers envelope.MessageID when set (regeneration targeting), then falls back
// to the last assistant message ID, then to fallback.
func ResolveMessageIDFromEnvelope(env ChatRequestEnvelope, fallback string) string {
	if env.MessageID != "" {
		return env.MessageID
	}
	return ResolveMessageID(env.Messages, fallback)
}

// EnvelopePartType identifies the kind of content in an EnvelopePart.
type EnvelopePartType string

const (
	EnvelopePartTypeText         EnvelopePartType = "text"
	EnvelopePartTypeImage        EnvelopePartType = "image"
	EnvelopePartTypeFile         EnvelopePartType = "file"
	EnvelopePartTypeToolInvocation EnvelopePartType = "tool-invocation"
)

// EnvelopePartUnion holds one content part inside an EnvelopeMessage.
// Only the fields matching Type are populated.
type EnvelopePartUnion struct {
	Type EnvelopePartType `json:"type"`

	// Text is set when Type == "text".
	Text string `json:"text,omitempty"`

	// URL is set when Type == "image" or "file".
	// May be a remote URL or a data: URI.
	URL string `json:"url,omitempty"`

	// MediaType is the MIME type for image/file parts (e.g. "image/png", "application/pdf").
	MediaType string `json:"mediaType,omitempty"`

	// Name is the original filename for file parts.
	Name string `json:"name,omitempty"`

	// FileID is a provider-specific file identifier (e.g. OpenAI "file-abc123").
	// When set, URL is ignored and the provider file ID is used directly.
	FileID string `json:"fileId,omitempty"`

	// Data holds inline binary content for image or file parts (base64-encoded in JSON).
	// When set, URL is ignored and the data is sent inline.
	Data []byte `json:"data,omitempty"`

	// Tool-invocation part fields (Type == "tool-invocation").
	// ToolCallID is the unique identifier of the tool call.
	ToolCallID string `json:"toolCallId,omitempty"`
	// ToolName is the function name of the tool.
	ToolName string `json:"toolName,omitempty"`
	// Input is the JSON-encoded tool call arguments.
	Input json.RawMessage `json:"input,omitempty"`
	// Output is the string result from the tool execution (present when State == "result").
	Output string `json:"output,omitempty"`
	// State is the tool invocation state: "partial-call", "call", "result", "error".
	State string `json:"state,omitempty"`
}
