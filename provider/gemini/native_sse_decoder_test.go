package gemini

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

func nativeStreamFromString(s string) io.ReadCloser {
	return io.NopCloser(strings.NewReader(s))
}

func collectNativeEvents(body io.ReadCloser) []ai.StreamEvent {
	ch := make(chan ai.StreamEvent, 128)
	decodeNativeSSEStream(context.Background(), body, ch)
	var events []ai.StreamEvent
	for e := range ch {
		events = append(events, e)
	}
	return events
}

func TestDecodeNativeSSE_SimpleTextStream(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},"modelVersion":"gemini-2.5-flash"}

data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	var textDeltas []string
	usageCount := 0
	finishCount := 0
	var lastFinish ai.StreamEvent
	for _, e := range events {
		switch e.Type {
		case ai.StreamEventTextDelta:
			textDeltas = append(textDeltas, e.TextDelta)
		case ai.StreamEventUsage:
			usageCount++
		case ai.StreamEventFinish:
			finishCount++
			lastFinish = e
		}
	}
	if len(textDeltas) != 2 {
		t.Errorf("expected 2 text deltas, got %d", len(textDeltas))
	}
	if textDeltas[0] != "Hello" || textDeltas[1] != " world" {
		t.Errorf("unexpected text deltas: %v", textDeltas)
	}
	if usageCount != 2 {
		t.Errorf("expected 2 usage events, got %d", usageCount)
	}
	if finishCount != 1 {
		t.Errorf("expected 1 finish event, got %d", finishCount)
	}
	if lastFinish.FinishReason != ai.FinishReasonStop {
		t.Errorf("expected stop, got %s", lastFinish.FinishReason)
	}
	if lastFinish.RawFinishReason != "STOP" {
		t.Errorf("expected raw STOP, got %s", lastFinish.RawFinishReason)
	}
}

func TestDecodeNativeSSE_ReasoningThought(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"text":"Let me think...","thought":true}],"role":"model"},"index":0}],"modelVersion":"gemini-2.5-flash"}

data: {"candidates":[{"content":{"parts":[{"text":"The answer is 42."}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":10,"totalTokenCount":15,"thoughtsTokenCount":8},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	reasoningCount := 0
	textCount := 0
	var usageEvent *ai.StreamEvent
	for _, e := range events {
		switch e.Type {
		case ai.StreamEventReasoningDelta:
			reasoningCount++
			if e.TextDelta != "Let me think..." {
				t.Errorf("unexpected reasoning text: %q", e.TextDelta)
			}
		case ai.StreamEventTextDelta:
			textCount++
		case ai.StreamEventUsage:
			ev := e
			usageEvent = &ev
		}
	}
	if reasoningCount != 1 {
		t.Errorf("expected 1 reasoning delta, got %d", reasoningCount)
	}
	if textCount != 1 {
		t.Errorf("expected 1 text delta, got %d", textCount)
	}
	if usageEvent == nil {
		t.Fatal("expected usage event")
	}
	if usageEvent.Usage.ReasoningTokens != 8 {
		t.Errorf("expected 8 reasoning tokens, got %d", usageEvent.Usage.ReasoningTokens)
	}
}

func TestDecodeNativeSSE_ToolCall(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"NYC","unit":"celsius"}}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":15,"totalTokenCount":35},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	toolCount := 0
	var toolEvent ai.StreamEvent
	var finishEvent ai.StreamEvent
	for _, e := range events {
		switch e.Type {
		case ai.StreamEventToolCallDelta:
			toolCount++
			toolEvent = e
		case ai.StreamEventFinish:
			finishEvent = e
		}
	}
	if toolCount != 1 {
		t.Fatalf("expected 1 tool call delta, got %d", toolCount)
	}
	if toolEvent.ToolCallName != "get_weather" {
		t.Errorf("expected tool name get_weather, got %q", toolEvent.ToolCallName)
	}
	if toolEvent.ToolCallID != "call_0" {
		t.Errorf("expected tool ID call_0, got %q", toolEvent.ToolCallID)
	}
	if toolEvent.ToolCallIndex != 0 {
		t.Errorf("expected tool index 0, got %d", toolEvent.ToolCallIndex)
	}
	// STOP + tool calls → tool_calls finish reason
	if finishEvent.FinishReason != ai.FinishReasonToolCalls {
		t.Errorf("expected tool_calls finish, got %s", finishEvent.FinishReason)
	}
	if finishEvent.RawFinishReason != "STOP" {
		t.Errorf("expected raw STOP, got %s", finishEvent.RawFinishReason)
	}
}

func TestDecodeNativeSSE_UsageMapping(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30,"thoughtsTokenCount":5,"cachedContentTokenCount":3},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	var usage *ai.Usage
	for _, e := range events {
		if e.Type == ai.StreamEventUsage {
			usage = e.Usage
		}
	}
	if usage == nil {
		t.Fatal("expected usage event")
	}
	if usage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 20 {
		t.Errorf("expected 20 completion tokens, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 30 {
		t.Errorf("expected 30 total tokens, got %d", usage.TotalTokens)
	}
	if usage.ReasoningTokens != 5 {
		t.Errorf("expected 5 reasoning tokens, got %d", usage.ReasoningTokens)
	}
}

func TestDecodeNativeSSE_GroundingSources(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"text":"According to sources..."}],"role":"model"},"finishReason":"STOP","index":0,"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com","title":"Example"}},{"web":{"uri":"https://other.com","title":"Other"}},{"web":{"uri":"https://example.com","title":"Example Dup"}},{"image":{"uri":"https://img.com/photo.jpg","title":"Photo"}},{"maps":{"uri":"https://maps.google.com/place","title":"Place"}}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	var sources []ai.Source
	for _, e := range events {
		if e.Type == ai.StreamEventSource {
			sources = append(sources, *e.Source)
		}
	}
	// 5 grounding chunks but https://example.com is duplicated → 4 unique sources
	if len(sources) != 4 {
		t.Fatalf("expected 4 sources (deduped), got %d", len(sources))
	}
	if sources[0].SourceType != "url" || sources[0].URL != "https://example.com" {
		t.Errorf("unexpected source 0: %+v", sources[0])
	}
	if sources[1].SourceType != "url" || sources[1].URL != "https://other.com" {
		t.Errorf("unexpected source 1: %+v", sources[1])
	}
	if sources[2].SourceType != "image" || sources[2].URL != "https://img.com/photo.jpg" {
		t.Errorf("unexpected source 2: %+v", sources[2])
	}
	if sources[3].SourceType != "maps" || sources[3].URL != "https://maps.google.com/place" {
		t.Errorf("unexpected source 3: %+v", sources[3])
	}
}

func TestDecodeNativeSSE_MaxTokensFinish(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"text":"truncated output"}],"role":"model"},"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":100,"totalTokenCount":110},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventFinish {
			if e.FinishReason != ai.FinishReasonLength {
				t.Errorf("expected length, got %s", e.FinishReason)
			}
			if e.RawFinishReason != "MAX_TOKENS" {
				t.Errorf("expected raw MAX_TOKENS, got %s", e.RawFinishReason)
			}
			return
		}
	}
	t.Error("no finish event found")
}

func TestDecodeNativeSSE_SafetyFinish(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"SAFETY","index":0}],"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventFinish {
			if e.FinishReason != ai.FinishReasonContentFilter {
				t.Errorf("expected content_filter, got %s", e.FinishReason)
			}
			if e.RawFinishReason != "SAFETY" {
				t.Errorf("expected raw SAFETY, got %s", e.RawFinishReason)
			}
			return
		}
	}
	t.Error("no finish event found")
}

func TestDecodeNativeSSE_EmptyStream(t *testing.T) {
	events := collectNativeEvents(nativeStreamFromString(""))
	if len(events) != 0 {
		t.Errorf("expected 0 events from empty stream, got %d", len(events))
	}
}

func TestDecodeNativeSSE_ContentAndFinishSameChunk(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"text":"final answer"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	// Verify ordering: text → usage → finish
	var types []ai.StreamEventType
	for _, e := range events {
		types = append(types, e.Type)
	}
	expected := []ai.StreamEventType{
		ai.StreamEventTextDelta,
		ai.StreamEventUsage,
		ai.StreamEventFinish,
	}
	if len(types) != len(expected) {
		t.Fatalf("expected %d events, got %d: %v", len(expected), len(types), types)
	}
	for i, et := range expected {
		if types[i] != et {
			t.Errorf("event %d: expected type %d, got %d", i, et, types[i])
		}
	}
}

func TestDecodeNativeSSE_MultipleToolCalls(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"golang"}}},{"functionCall":{"name":"translate","args":{"text":"hello","to":"es"}}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":25,"totalTokenCount":40},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	var tools []ai.StreamEvent
	var finishEvent ai.StreamEvent
	for _, e := range events {
		switch e.Type {
		case ai.StreamEventToolCallDelta:
			tools = append(tools, e)
		case ai.StreamEventFinish:
			finishEvent = e
		}
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tool call deltas, got %d", len(tools))
	}
	if tools[0].ToolCallName != "search" || tools[0].ToolCallID != "call_0" || tools[0].ToolCallIndex != 0 {
		t.Errorf(
			"unexpected tool 0: name=%q id=%q idx=%d",
			tools[0].ToolCallName,
			tools[0].ToolCallID,
			tools[0].ToolCallIndex,
		)
	}
	if tools[1].ToolCallName != "translate" || tools[1].ToolCallID != "call_1" || tools[1].ToolCallIndex != 1 {
		t.Errorf(
			"unexpected tool 1: name=%q id=%q idx=%d",
			tools[1].ToolCallName,
			tools[1].ToolCallID,
			tools[1].ToolCallIndex,
		)
	}
	if finishEvent.FinishReason != ai.FinishReasonToolCalls {
		t.Errorf("expected tool_calls finish, got %s", finishEvent.FinishReason)
	}
}

func TestDecodeNativeSSE_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	sse := `data: {"candidates":[{"content":{"parts":[{"text":"text"}],"role":"model"},"index":0}]}
`
	ch := make(chan ai.StreamEvent, 128)
	decodeNativeSSEStream(ctx, nativeStreamFromString(sse), ch)

	hasError := false
	for e := range ch {
		if e.Type == ai.StreamEventError {
			hasError = true
		}
	}
	if !hasError {
		t.Error("expected error event on context cancellation")
	}
}

func TestDecodeNativeSSE_ProviderMetadataOnFinish(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"text":"grounded"}],"role":"model"},"finishReason":"STOP","index":0,"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com","title":"Example"}}]},"safetyRatings":[{"category":"HARM_CATEGORY_HARASSMENT","probability":"NEGLIGIBLE"}]}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventFinish {
			if e.ProviderMetadata == nil {
				t.Fatal("expected provider metadata on finish")
			}
			google, ok := e.ProviderMetadata["google"].(map[string]any)
			if !ok {
				t.Fatal("expected google metadata")
			}
			if _, ok := google["groundingMetadata"]; !ok {
				t.Error("expected groundingMetadata in google metadata")
			}
			if _, ok := google["safetyRatings"]; !ok {
				t.Error("expected safetyRatings in google metadata")
			}
			return
		}
	}
	t.Error("no finish event found")
}

func TestDecodeNativeSSE_MalformedFunctionCallFinish(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"MALFORMED_FUNCTION_CALL","index":0}],"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventFinish {
			if e.FinishReason != ai.FinishReasonError {
				t.Errorf("expected error finish, got %s", e.FinishReason)
			}
			return
		}
	}
	t.Error("no finish event found")
}

func TestDecodeNativeSSE_ThoughtSignature(t *testing.T) {
	sse := `data: {"candidates":[{"content":{"parts":[{"text":"thinking...","thought":true,"thoughtSignature":"sig123"}],"role":"model"},"index":0}]}

data: {"candidates":[{"content":{"parts":[{"text":"done"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.5-flash"}
`
	events := collectNativeEvents(nativeStreamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventReasoningDelta {
			if e.ThoughtSignature != "sig123" {
				t.Errorf("expected sig123, got %q", e.ThoughtSignature)
			}
			return
		}
	}
	t.Error("no reasoning delta found")
}
