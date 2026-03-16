package gemini

import (
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// groundingSSE builds a minimal SSE fixture with provider_metadata containing groundingMetadata.
const groundingSSE = `data: {"choices":[{"delta":{"content":"The answer is 42."},"finish_reason":null}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}],"provider_metadata":{"google":{"groundingMetadata":{"webSearchQueries":["meaning of life"],"groundingChunks":[{"web":{"uri":"https://example.com/page1","title":"Example Page 1"}},{"web":{"uri":"https://example.com/page2","title":"Example Page 2"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":17,"text":"The answer is 42."},"groundingChunkIndices":[0,1],"confidenceScores":[0.95,0.88]}],"searchEntryPoint":{"renderedContent":"<html>search entry</html>"}}}}}
data: [DONE]
`

func TestGrounding_SourcesExtracted(t *testing.T) {
	events := collectEvents(streamFromString(groundingSSE))

	var sources []ai.StreamEvent
	for _, e := range events {
		if e.Type == ai.StreamEventSource {
			sources = append(sources, e)
		}
	}

	if len(sources) != 2 {
		t.Fatalf("expected 2 source events, got %d", len(sources))
	}
	if sources[0].Source.URL != "https://example.com/page1" {
		t.Errorf("unexpected first source URL: %q", sources[0].Source.URL)
	}
	if sources[0].Source.Title != "Example Page 1" {
		t.Errorf("unexpected first source title: %q", sources[0].Source.Title)
	}
	if sources[1].Source.URL != "https://example.com/page2" {
		t.Errorf("unexpected second source URL: %q", sources[1].Source.URL)
	}
	if sources[0].Source.SourceType != "url" {
		t.Errorf("expected SourceType=url, got %q", sources[0].Source.SourceType)
	}
}

func TestGrounding_ProviderMetadataOnFinish(t *testing.T) {
	events := collectEvents(streamFromString(groundingSSE))

	var finishEvent *ai.StreamEvent
	for i := range events {
		if events[i].Type == ai.StreamEventFinish && events[i].RawFinishReason == "stop" {
			finishEvent = &events[i]
			break
		}
	}
	if finishEvent == nil {
		t.Fatal("expected a finish event with RawFinishReason=stop")
	}
	if finishEvent.ProviderMetadata == nil {
		t.Fatal("expected ProviderMetadata to be set on finish event")
	}
	google, ok := finishEvent.ProviderMetadata["google"].(map[string]any)
	if !ok {
		t.Fatalf("expected provider_metadata.google to be map, got %T", finishEvent.ProviderMetadata["google"])
	}
	gm, ok := google["groundingMetadata"].(map[string]any)
	if !ok {
		t.Fatalf("expected groundingMetadata to be map, got %T", google["groundingMetadata"])
	}
	queries, ok := gm["webSearchQueries"].([]any)
	if !ok || len(queries) == 0 {
		t.Errorf("expected webSearchQueries in groundingMetadata, got %v", gm["webSearchQueries"])
	}
	if queries[0] != "meaning of life" {
		t.Errorf("unexpected webSearchQuery: %q", queries[0])
	}
}

func TestGrounding_Deduplication(t *testing.T) {
	// Same URL appears in two chunks — should only emit one source.
	sse := `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"provider_metadata":{"google":{"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://dup.example.com","title":"Dup"}},{"web":{"uri":"https://dup.example.com","title":"Dup Again"}}]}}}}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	var sources []ai.StreamEvent
	for _, e := range events {
		if e.Type == ai.StreamEventSource {
			sources = append(sources, e)
		}
	}
	if len(sources) != 1 {
		t.Errorf("expected 1 source after dedup, got %d", len(sources))
	}
}

func TestGrounding_NoMetadata(t *testing.T) {
	// Regular response without grounding metadata should produce no source events.
	sse := `data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventSource {
			t.Error("unexpected source event in non-grounding response")
		}
	}
}

func TestGrounding_EmptyGroundingChunks(t *testing.T) {
	sse := `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"provider_metadata":{"google":{"groundingMetadata":{"webSearchQueries":["test"],"groundingChunks":[]}}}}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventSource {
			t.Error("unexpected source event from empty groundingChunks")
		}
	}
}

func TestGrounding_MultipleChunksWithSameMetadata(t *testing.T) {
	// If the same grounding chunk URI appears in two separate SSE chunks, dedup across chunks.
	sse := `data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}],"provider_metadata":{"google":{"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://crosschunk.example.com","title":"Cross"}}]}}}}
data: {"choices":[{"delta":{},"finish_reason":"stop"}],"provider_metadata":{"google":{"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://crosschunk.example.com","title":"Cross Again"}}]}}}}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	var sources []ai.StreamEvent
	for _, e := range events {
		if e.Type == ai.StreamEventSource {
			sources = append(sources, e)
		}
	}
	if len(sources) != 1 {
		t.Errorf("expected 1 unique source across chunks, got %d", len(sources))
	}
}
