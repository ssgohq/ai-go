// Package openaichat implements the shared OpenAI chat completions codec
// used by Gemini and generic OpenAI-compatible providers.
package openaichat

import (
	"strings"

	"github.com/ssgohq/ai-go/ai"
)

// MapFinishReason converts a raw chat-completions finish_reason string into an ai.FinishReason.
func MapFinishReason(s string) ai.FinishReason {
	switch strings.ToLower(s) {
	case "stop":
		return ai.FinishReasonStop
	case "tool_calls":
		return ai.FinishReasonToolCalls
	case "length":
		return ai.FinishReasonLength
	case "content_filter":
		return ai.FinishReasonContentFilter
	default:
		return ai.FinishReasonUnknown
	}
}
