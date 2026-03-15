package uistream

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
}

// EnvelopeMessage is a single message inside ChatRequestEnvelope.
// Role is one of: "user", "assistant", "system".
// Parts carry the actual content; for simple text messages a single TextEnvelopePart suffices.
type EnvelopeMessage struct {
	Role  string              `json:"role"`
	Parts []EnvelopePartUnion `json:"parts,omitempty"`

	// Content is the flat string shorthand for a single-text-part message.
	// If Parts is non-empty, Content is ignored.
	Content string `json:"content,omitempty"`
}

// EnvelopePartType identifies the kind of content in an EnvelopePart.
type EnvelopePartType string

const (
	EnvelopePartTypeText  EnvelopePartType = "text"
	EnvelopePartTypeImage EnvelopePartType = "image"
	EnvelopePartTypeFile  EnvelopePartType = "file"
)

// EnvelopePartUnion holds one content part inside an EnvelopeMessage.
// Only the field matching Type is populated.
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
}
