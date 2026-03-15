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
	ContentPartTypeText     ContentPartType = "text"
	ContentPartTypeImageURL ContentPartType = "image_url"
	ContentPartTypeFile     ContentPartType = "file"
	ContentPartTypeToolCall ContentPartType = "tool_call"
	ContentPartTypeToolResult ContentPartType = "tool_result"
)

// ContentPart is a single part of a message (text, image, file, or tool call/result).
type ContentPart struct {
	Type ContentPartType

	// Text is set when Type == ContentPartTypeText.
	Text string

	// ImageURL is set when Type == ContentPartTypeImageURL (supports data: URLs).
	ImageURL string

	// FileURL is set when Type == ContentPartTypeFile.
	FileURL  string
	MimeType string

	// ToolCall is set when Type == ContentPartTypeToolCall.
	ToolCallID   string
	ToolCallName string
	ToolCallArgs json.RawMessage

	// ToolResult is set when Type == ContentPartTypeToolResult.
	ToolResultID     string
	ToolResultName   string
	ToolResultOutput string
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
