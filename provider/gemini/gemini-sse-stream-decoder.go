package gemini

import (
	"context"
	"io"

	"github.com/ssgohq/ai-go/ai"
	"github.com/ssgohq/ai-go/provider/internal/openaichat"
)

// decodeSSEStream reads SSE lines from body and emits normalized ai.StreamEvents onto ch.
// It delegates to the shared openaichat decoder.
func decodeSSEStream(ctx context.Context, body io.ReadCloser, ch chan<- ai.StreamEvent) {
	openaichat.DecodeSSEStream(ctx, body, ch, openaichat.SSEDecodeParams{
		ProviderName: "gemini",
	})
}
