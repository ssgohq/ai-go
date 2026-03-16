package gemini

import (
	"context"
	"io"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

// decodeSSEStream reads SSE lines from body and emits normalized ai.StreamEvents onto ch.
// It delegates to the shared openaichat decoder with Gemini-specific extractors.
func decodeSSEStream(ctx context.Context, body io.ReadCloser, ch chan<- ai.StreamEvent) {
	seen := make(map[string]bool)
	openaichat.DecodeSSEStream(ctx, body, ch, openaichat.SSEDecodeParams{
		ProviderName:      "gemini",
		MetadataExtractor: extractGroundingMetadata,
		SourceExtractor:   deduplicatedSourceExtractor(seen),
	})
}

// deduplicatedSourceExtractor returns a SourceExtractor closure that deduplicates sources by URL.
func deduplicatedSourceExtractor(seen map[string]bool) func(openaichat.StreamChunk) []ai.Source {
	return func(chunk openaichat.StreamChunk) []ai.Source {
		gm := groundingMetadataFromChunk(chunk)
		if gm == nil {
			return nil
		}
		chunks, ok := gm["groundingChunks"].([]any)
		if !ok {
			return nil
		}
		var sources []ai.Source
		for _, c := range chunks {
			cm, ok := c.(map[string]any)
			if !ok {
				continue
			}
			web, ok := cm["web"].(map[string]any)
			if !ok {
				continue
			}
			uri, _ := web["uri"].(string)
			title, _ := web["title"].(string)
			if uri == "" || seen[uri] {
				continue
			}
			seen[uri] = true
			sources = append(sources, ai.Source{
				SourceType: "url",
				URL:        uri,
				Title:      title,
			})
		}
		return sources
	}
}

// extractGroundingMetadata returns the full groundingMetadata wrapped under "google" key,
// suitable for use as ProviderMetadata on finish events.
func extractGroundingMetadata(chunk openaichat.StreamChunk) map[string]any {
	gm := groundingMetadataFromChunk(chunk)
	if gm == nil {
		return nil
	}
	return map[string]any{
		"google": map[string]any{
			"groundingMetadata": gm,
		},
	}
}

// groundingMetadataFromChunk digs into provider_metadata.google.groundingMetadata.
func groundingMetadataFromChunk(chunk openaichat.StreamChunk) map[string]any {
	if chunk.ProviderMetadata == nil {
		return nil
	}
	google, ok := chunk.ProviderMetadata["google"].(map[string]any)
	if !ok {
		return nil
	}
	gm, ok := google["groundingMetadata"].(map[string]any)
	if !ok {
		return nil
	}
	return gm
}
