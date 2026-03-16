package gemini

import (
	"context"
	"io"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

// decodeSSEStream reads SSE lines from body and emits normalized ai.StreamEvents onto ch.
// It delegates to the shared openaichat decoder with Gemini-specific extractors.
// Metadata (groundingMetadata, safetyRatings, urlContextMetadata) is accumulated across
// all chunks so that the finish event carries the last-seen values even if they arrived
// in a non-final chunk.
func decodeSSEStream(ctx context.Context, body io.ReadCloser, ch chan<- ai.StreamEvent) {
	seen := make(map[string]bool)

	// lastMeta holds the most-recently-seen google metadata across all chunks.
	var lastMeta map[string]any

	metaExtractor := func(chunk openaichat.StreamChunk) map[string]any {
		m := buildGoogleMetadata(chunk)
		if m != nil {
			lastMeta = m
		}
		if lastMeta == nil {
			return nil
		}
		return map[string]any{"google": lastMeta}
	}

	sourceExtractor := deduplicatedSourceExtractor(seen)

	openaichat.DecodeSSEStream(ctx, body, ch, openaichat.SSEDecodeParams{
		ProviderName:      "gemini",
		MetadataExtractor: metaExtractor,
		SourceExtractor:   sourceExtractor,
	})
}

// deduplicatedSourceExtractor returns a SourceExtractor closure that deduplicates sources
// by their key (URL for web, URI for other types) across all stream chunks.
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
			if src := extractWebSource(cm, seen); src != nil {
				sources = append(sources, *src)
				continue
			}
			if src := extractRetrievedContextSource(cm, seen); src != nil {
				sources = append(sources, *src)
				continue
			}
			if src := extractImageSource(cm, seen); src != nil {
				sources = append(sources, *src)
				continue
			}
			if src := extractMapsSource(cm, seen); src != nil {
				sources = append(sources, *src)
			}
		}
		return sources
	}
}

// extractWebSource extracts a "url" source from a grounding chunk's "web" field.
func extractWebSource(cm map[string]any, seen map[string]bool) *ai.Source {
	web, ok := cm["web"].(map[string]any)
	if !ok {
		return nil
	}
	uri, ok := web["uri"].(string)
	if !ok || uri == "" {
		return nil
	}
	if seen[uri] {
		return nil
	}
	seen[uri] = true
	title, _ := web["title"].(string)
	return &ai.Source{
		SourceType: "url",
		URL:        uri,
		Title:      title,
	}
}

// extractRetrievedContextSource extracts a "retrieved-context" source from a grounding chunk.
func extractRetrievedContextSource(cm map[string]any, seen map[string]bool) *ai.Source {
	rc, ok := cm["retrievedContext"].(map[string]any)
	if !ok {
		return nil
	}
	uri, ok := rc["uri"].(string)
	if !ok || uri == "" {
		return nil
	}
	if seen[uri] {
		return nil
	}
	seen[uri] = true
	title, _ := rc["title"].(string)
	return &ai.Source{
		SourceType: "retrieved-context",
		URL:        uri,
		Title:      title,
	}
}

// extractImageSource extracts an "image" source from a grounding chunk's "image" field.
func extractImageSource(cm map[string]any, seen map[string]bool) *ai.Source {
	img, ok := cm["image"].(map[string]any)
	if !ok {
		return nil
	}
	uri, _ := img["uri"].(string)
	if uri == "" {
		// Some image chunks may only have other fields; use a placeholder key.
		uri = "image-chunk"
	}
	key := "image:" + uri
	if seen[key] {
		return nil
	}
	seen[key] = true
	title, _ := img["title"].(string)
	return &ai.Source{
		SourceType:       "image",
		URL:              uri,
		Title:            title,
		ProviderMetadata: map[string]any{"image": img},
	}
}

// extractMapsSource extracts a "maps" source from a grounding chunk's "maps" field.
func extractMapsSource(cm map[string]any, seen map[string]bool) *ai.Source {
	maps, ok := cm["maps"].(map[string]any)
	if !ok {
		return nil
	}
	uri, _ := maps["uri"].(string)
	if uri == "" {
		uri = "maps-chunk"
	}
	key := "maps:" + uri
	if seen[key] {
		return nil
	}
	seen[key] = true
	title, _ := maps["title"].(string)
	return &ai.Source{
		SourceType:       "maps",
		URL:              uri,
		Title:            title,
		ProviderMetadata: map[string]any{"maps": maps},
	}
}

// buildGoogleMetadata assembles the google-namespaced metadata map from a chunk.
// It includes groundingMetadata, safetyRatings, and urlContextMetadata when present.
// Returns nil if none of the fields are found.
func buildGoogleMetadata(chunk openaichat.StreamChunk) map[string]any {
	if chunk.ProviderMetadata == nil {
		return nil
	}
	google, ok := chunk.ProviderMetadata["google"].(map[string]any)
	if !ok {
		return nil
	}

	result := make(map[string]any)
	if gm, ok := google["groundingMetadata"].(map[string]any); ok {
		result["groundingMetadata"] = gm
	}
	if sr, ok := google["safetyRatings"]; ok {
		result["safetyRatings"] = sr
	}
	if uc, ok := google["urlContextMetadata"]; ok {
		result["urlContextMetadata"] = uc
	}

	if len(result) == 0 {
		return nil
	}
	return result
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
