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
func ToAIContentParts(parts []EnvelopePartUnion) []ai.ContentPart {
	result := make([]ai.ContentPart, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case EnvelopePartTypeText:
			result = append(result, ai.TextPart(p.Text))
		case EnvelopePartTypeImage:
			result = append(result, ai.ImageURLPart(p.URL))
		case EnvelopePartTypeFile:
			result = append(result, ai.FilePart(p.URL, p.MediaType))
		}
	}
	return result
}
