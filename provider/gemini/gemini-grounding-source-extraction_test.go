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

func TestGrounding_MixedSourceTypes(t *testing.T) {
	// Fixture with web + retrievedContext + image chunks.
	sse := `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"provider_metadata":{"google":{"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://web.example.com","title":"Web Result"}},{"retrievedContext":{"uri":"https://doc.example.com/doc1","title":"Doc One"}},{"image":{"uri":"https://img.example.com/photo.jpg","title":"Photo"}}]}}}}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	var sources []ai.StreamEvent
	for _, e := range events {
		if e.Type == ai.StreamEventSource {
			sources = append(sources, e)
		}
	}

	if len(sources) != 3 {
		t.Fatalf("expected 3 source events (web+retrievedContext+image), got %d", len(sources))
	}

	byType := make(map[string]*ai.Source)
	for i := range sources {
		byType[sources[i].Source.SourceType] = sources[i].Source
	}

	webSrc, ok := byType["url"]
	if !ok {
		t.Fatal("expected a source with SourceType=url")
	}
	if webSrc.URL != "https://web.example.com" {
		t.Errorf("unexpected web URL: %q", webSrc.URL)
	}
	if webSrc.Title != "Web Result" {
		t.Errorf("unexpected web title: %q", webSrc.Title)
	}

	rcSrc, ok := byType["retrieved-context"]
	if !ok {
		t.Fatal("expected a source with SourceType=retrieved-context")
	}
	if rcSrc.URL != "https://doc.example.com/doc1" {
		t.Errorf("unexpected retrieved-context URL: %q", rcSrc.URL)
	}
	if rcSrc.Title != "Doc One" {
		t.Errorf("unexpected retrieved-context title: %q", rcSrc.Title)
	}

	imgSrc, ok := byType["image"]
	if !ok {
		t.Fatal("expected a source with SourceType=image")
	}
	if imgSrc.URL != "https://img.example.com/photo.jpg" {
		t.Errorf("unexpected image URL: %q", imgSrc.URL)
	}
	if imgSrc.Title != "Photo" {
		t.Errorf("unexpected image title: %q", imgSrc.Title)
	}
}

func TestGrounding_SafetyRatingsInMetadata(t *testing.T) {
	sse := `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"provider_metadata":{"google":{"groundingMetadata":{"webSearchQueries":["test"]},"safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"NEGLIGIBLE"}]}}}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	var finishEvent *ai.StreamEvent
	for i := range events {
		if events[i].Type == ai.StreamEventFinish {
			finishEvent = &events[i]
			break
		}
	}
	if finishEvent == nil {
		t.Fatal("expected a finish event")
	}
	if finishEvent.ProviderMetadata == nil {
		t.Fatal("expected ProviderMetadata on finish event")
	}

	google, ok := finishEvent.ProviderMetadata["google"].(map[string]any)
	if !ok {
		t.Fatalf("expected google metadata map, got %T", finishEvent.ProviderMetadata["google"])
	}

	sr, ok := google["safetyRatings"]
	if !ok {
		t.Fatal("expected safetyRatings in google metadata")
	}
	ratings, ok := sr.([]any)
	if !ok || len(ratings) == 0 {
		t.Fatalf("expected safetyRatings to be a non-empty array, got %T %v", sr, sr)
	}
	first, ok := ratings[0].(map[string]any)
	if !ok {
		t.Fatalf("expected first safetyRating to be a map, got %T", ratings[0])
	}
	if first["category"] != "HARM_CATEGORY_HATE_SPEECH" {
		t.Errorf("unexpected safety rating category: %v", first["category"])
	}
}

func TestGrounding_URLContextMetadata(t *testing.T) {
	sse := `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"provider_metadata":{"google":{"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://example.com","urlRetrievalStatus":"URL_RETRIEVAL_STATUS_SUCCESS"}]}}}}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	var finishEvent *ai.StreamEvent
	for i := range events {
		if events[i].Type == ai.StreamEventFinish {
			finishEvent = &events[i]
			break
		}
	}
	if finishEvent == nil {
		t.Fatal("expected a finish event")
	}
	if finishEvent.ProviderMetadata == nil {
		t.Fatal("expected ProviderMetadata on finish event")
	}

	google, ok := finishEvent.ProviderMetadata["google"].(map[string]any)
	if !ok {
		t.Fatalf("expected google metadata map, got %T", finishEvent.ProviderMetadata["google"])
	}

	uc, ok := google["urlContextMetadata"]
	if !ok {
		t.Fatal("expected urlContextMetadata in google metadata")
	}
	ucMap, ok := uc.(map[string]any)
	if !ok {
		t.Fatalf("expected urlContextMetadata to be a map, got %T", uc)
	}
	if _, ok := ucMap["urlMetadata"]; !ok {
		t.Error("expected urlMetadata key in urlContextMetadata")
	}
}

func TestGrounding_MetadataInNonFinalChunk(t *testing.T) {
	// groundingMetadata arrives in a non-final chunk (before finish_reason).
	// The finish event should still carry the accumulated metadata.
	sse := `data: {"choices":[{"delta":{"content":"answer"},"finish_reason":null}],"provider_metadata":{"google":{"groundingMetadata":{"webSearchQueries":["non-final query"],"groundingChunks":[{"web":{"uri":"https://early.example.com","title":"Early Source"}}]}}}}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
`
	events := collectEvents(streamFromString(sse))

	// Verify source was extracted from the non-final chunk.
	var sources []ai.StreamEvent
	var finishEvent *ai.StreamEvent
	for i := range events {
		switch events[i].Type {
		case ai.StreamEventSource:
			sources = append(sources, events[i])
		case ai.StreamEventFinish:
			if events[i].RawFinishReason == "stop" {
				finishEvent = &events[i]
			}
		}
	}

	if len(sources) != 1 {
		t.Fatalf("expected 1 source from non-final chunk, got %d", len(sources))
	}
	if sources[0].Source.URL != "https://early.example.com" {
		t.Errorf("unexpected source URL: %q", sources[0].Source.URL)
	}

	if finishEvent == nil {
		t.Fatal("expected a finish event with RawFinishReason=stop")
	}
	if finishEvent.ProviderMetadata == nil {
		t.Fatal("expected ProviderMetadata to be carried on finish event from earlier chunk")
	}

	google, ok := finishEvent.ProviderMetadata["google"].(map[string]any)
	if !ok {
		t.Fatalf("expected google metadata map on finish, got %T", finishEvent.ProviderMetadata["google"])
	}
	gm, ok := google["groundingMetadata"].(map[string]any)
	if !ok {
		t.Fatalf("expected groundingMetadata on finish, got %T", google["groundingMetadata"])
	}
	queries, ok := gm["webSearchQueries"].([]any)
	if !ok || len(queries) == 0 {
		t.Errorf("expected webSearchQueries in accumulated groundingMetadata, got %v", gm["webSearchQueries"])
	}
	if queries[0] != "non-final query" {
		t.Errorf("unexpected webSearchQuery: %q", queries[0])
	}
}
