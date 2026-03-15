package gemini

import (
	"github.com/ssgohq/ai-go/ai"
	"github.com/ssgohq/ai-go/provider/internal/openaichat"
)

// encodeRequest converts an ai.LanguageModelRequest into a shared ChatRequest
// using the Gemini-specific sanitization hook and stream-usage flag.
func encodeRequest(
	modelID string,
	req ai.LanguageModelRequest,
	streaming bool,
) (openaichat.ChatRequest, error) {
	return openaichat.EncodeRequest(openaichat.EncodeRequestParams{
		ModelID:            modelID,
		SanitizeTools:      sanitizeToolSchemas,
		IncludeStreamUsage: streaming,
	}, req, streaming)
}
