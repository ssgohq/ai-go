// Package ai provides the public API surface for the ai-go SDK.
package ai

import "encoding/json"

// Role identifies the author of a message in a conversation.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentPartType identifies the kind of content in a message part.
type ContentPartType string

const (
	// ContentPartTypeText is a plain-text part.
	ContentPartTypeText ContentPartType = "text"
	// ContentPartTypeImageURL is an image referenced by URL or data URI.
	ContentPartTypeImageURL ContentPartType = "image_url"
	// ContentPartTypeFile is a file referenced by URL or data URI.
	ContentPartTypeFile ContentPartType = "file"
	// ContentPartTypeToolCall is a model-issued tool call (assistant turn only).
	ContentPartTypeToolCall ContentPartType = "tool_call"
	// ContentPartTypeToolResult is the result of a tool execution (tool turn only).
	ContentPartTypeToolResult ContentPartType = "tool_result"
	// ContentPartTypeReasoning carries prior reasoning text for history replay
	// (e.g. Claude extended thinking). Used in assistant messages when replaying
	// multi-step conversations that included a reasoning block.
	ContentPartTypeReasoning ContentPartType = "reasoning"
)

// ContentPart is a single part of a message (text, image, file, tool call/result, or reasoning).
// Only the fields matching the active Type are populated; all others are zero.
type ContentPart struct {
	// Type identifies which fields below are populated.
	Type ContentPartType

	// Text is set when Type == ContentPartTypeText.
	Text string

	// ImageURL is set when Type == ContentPartTypeImageURL (supports data: URLs).
	ImageURL string

	// FileURL is the URL or data URI when Type == ContentPartTypeFile.
	FileURL string
	// MimeType is the MIME type of the file when Type == ContentPartTypeFile.
	MimeType string

	// ToolCallID is the unique ID for the tool call when Type == ContentPartTypeToolCall.
	ToolCallID string
	// ToolCallName is the tool function name when Type == ContentPartTypeToolCall.
	ToolCallName string
	// ToolCallArgs is the JSON-encoded arguments when Type == ContentPartTypeToolCall.
	ToolCallArgs json.RawMessage
	// ThoughtSignature is the Gemini thought signature for multi-turn tool calls.
	ThoughtSignature string

	// ToolResultID matches the ToolCallID that this result answers (Type == ContentPartTypeToolResult).
	ToolResultID string
	// ToolResultName is the name of the tool that produced this result (Type == ContentPartTypeToolResult).
	ToolResultName string
	// ToolResultOutput is the string result from the tool execution (Type == ContentPartTypeToolResult).
	ToolResultOutput string

	// ReasoningText holds the reasoning / thinking text when Type == ContentPartTypeReasoning.
	ReasoningText string
}

// TextPart constructs a text ContentPart.
func TextPart(text string) ContentPart {
	return ContentPart{Type: ContentPartTypeText, Text: text}
}

// ImageURLPart constructs an image ContentPart from a URL or data URI.
func ImageURLPart(url string) ContentPart {
	return ContentPart{Type: ContentPartTypeImageURL, ImageURL: url}
}

// FilePart constructs a file ContentPart.
func FilePart(url, mimeType string) ContentPart {
	return ContentPart{Type: ContentPartTypeFile, FileURL: url, MimeType: mimeType}
}

// ReasoningPart constructs a reasoning ContentPart for history replay.
// Use this when reconstructing assistant messages that included a reasoning block
// (e.g. Claude extended thinking) so that the model can continue from prior reasoning.
func ReasoningPart(reasoningText string) ContentPart {
	return ContentPart{Type: ContentPartTypeReasoning, ReasoningText: reasoningText}
}

// ToolCallPart constructs a tool-call ContentPart for assistant messages.
func ToolCallPart(id, name string, args json.RawMessage) ContentPart {
	return ContentPart{
		Type:         ContentPartTypeToolCall,
		ToolCallID:   id,
		ToolCallName: name,
		ToolCallArgs: args,
	}
}

// ToolResultPart constructs a tool-result ContentPart for tool messages.
func ToolResultPart(id, name, output string) ContentPart {
	return ContentPart{
		Type:             ContentPartTypeToolResult,
		ToolResultID:     id,
		ToolResultName:   name,
		ToolResultOutput: output,
	}
}

// Message is a single turn in a conversation.
type Message struct {
	Role    Role
	Content []ContentPart
}

// UserMessage creates a user message with a single text part.
func UserMessage(text string) Message {
	return Message{Role: RoleUser, Content: []ContentPart{TextPart(text)}}
}

// AssistantMessage creates an assistant message with a single text part.
func AssistantMessage(text string) Message {
	return Message{Role: RoleAssistant, Content: []ContentPart{TextPart(text)}}
}

// SystemMessage creates a system message with a single text part.
func SystemMessage(text string) Message {
	return Message{Role: RoleSystem, Content: []ContentPart{TextPart(text)}}
}
