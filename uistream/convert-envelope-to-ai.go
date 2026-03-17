package uistream

import "github.com/open-ai-sdk/ai-go/ai"

// ToAIMessages converts a slice of EnvelopeMessage to ai.Message values.
// If a message has Parts, they are converted via ToAIContentParts.
// Otherwise, Content is used as a single text part.
func ToAIMessages(msgs []EnvelopeMessage) []ai.Message {
	result := make([]ai.Message, len(msgs))
	for i, m := range msgs {
		var parts []ai.ContentPart
		if len(m.Parts) > 0 {
			parts = ToAIContentParts(m.Parts)
		} else {
			parts = []ai.ContentPart{ai.TextPart(m.Content)}
		}
		result[i] = ai.Message{
			Role:    ai.Role(m.Role),
			Content: parts,
		}
	}
	return result
}

// ToAIContentParts converts a slice of EnvelopePartUnion to ai.ContentPart values.
// Priority for image/file parts: FileID > Data > URL.
func ToAIContentParts(parts []EnvelopePartUnion) []ai.ContentPart {
	out := make([]ai.ContentPart, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case EnvelopePartTypeText:
			out = append(out, ai.TextPart(p.Text))
		case EnvelopePartTypeImage:
			switch {
			case p.FileID != "":
				out = append(out, ai.ImageFileIDPart(p.FileID))
			case len(p.Data) > 0:
				out = append(out, ai.ImageDataPart(p.Data, p.MediaType))
			default:
				out = append(out, ai.ImageURLPart(p.URL))
			}
		case EnvelopePartTypeFile:
			switch {
			case p.FileID != "":
				out = append(out, ai.FileIDPart(p.FileID, p.MediaType))
			case len(p.Data) > 0:
				out = append(out, ai.FileDataPart(p.Data, p.MediaType, p.Name))
			default:
				cp := ai.FilePart(p.URL, p.MediaType)
				cp.Filename = p.Name
				out = append(out, cp)
			}
		case EnvelopePartTypeToolInvocation:
			out = append(out, toolInvocationParts(p)...)
		}
	}
	return out
}

// toolInvocationParts converts a tool-invocation EnvelopePartUnion into the
// corresponding ai.ContentPart(s). Only finalized states ("call", "result")
// produce parts; partial/unknown states are silently skipped.
//
// For state "result", both a ToolCallPart (for assistant context) and a
// ToolResultPart (for tool-role context) are emitted. Callers that need role
// separation should filter by ContentPartType after conversion.
func toolInvocationParts(p EnvelopePartUnion) []ai.ContentPart {
	switch p.State {
	case "call", "partial-call":
		return []ai.ContentPart{ai.ToolCallPart(p.ToolCallID, p.ToolName, p.Input)}
	case "result":
		return []ai.ContentPart{
			ai.ToolCallPart(p.ToolCallID, p.ToolName, p.Input),
			ai.ToolResultPart(p.ToolCallID, p.ToolName, p.Output),
		}
	default:
		// "error" and unknown states: skip gracefully.
		return nil
	}
}
