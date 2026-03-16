package openai

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// helpers

func streamFromString(s string) io.ReadCloser {
	return io.NopCloser(strings.NewReader(s))
}

func collectEvents(body io.ReadCloser) []ai.StreamEvent {
	ch := make(chan ai.StreamEvent, 128)
	decodeResponsesSSEStream(context.Background(), body, ch)
	var out []ai.StreamEvent
	for e := range ch {
		out = append(out, e)
	}
	return out
}

// SSE decoder tests

func TestOpenAISSE_TextDelta(t *testing.T) {
	sse := `data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}
data: {"type":"response.output_text.delta","delta":"Hello"}
data: {"type":"response.output_text.delta","delta":" world"}
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}
`
	events := collectEvents(streamFromString(sse))

	textDeltas := 0
	usageCount := 0
	finishCount := 0
	for _, e := range events {
		switch e.Type {
		case ai.StreamEventTextDelta:
			textDeltas++
		case ai.StreamEventUsage:
			usageCount++
			if e.Usage.PromptTokens != 5 {
				t.Errorf("expected 5 input tokens, got %d", e.Usage.PromptTokens)
			}
		case ai.StreamEventFinish:
			finishCount++
			if e.FinishReason != ai.FinishReasonStop {
				t.Errorf("expected FinishReasonStop, got %q", e.FinishReason)
			}
			if e.RawFinishReason != "completed" {
				t.Errorf("expected RawFinishReason=completed, got %q", e.RawFinishReason)
			}
		}
	}
	if textDeltas != 2 {
		t.Errorf("expected 2 text deltas, got %d", textDeltas)
	}
	if usageCount != 1 {
		t.Errorf("expected 1 usage event, got %d", usageCount)
	}
	if finishCount != 1 {
		t.Errorf("expected 1 finish event, got %d", finishCount)
	}
}

func TestOpenAISSE_ReasoningDelta(t *testing.T) {
	sse := `data: {"type":"response.reasoning_summary_text.delta","delta":"thinking..."}
data: {"type":"response.completed","response":{"id":"resp_2","status":"completed"}}
`
	events := collectEvents(streamFromString(sse))

	hasReasoning := false
	for _, e := range events {
		if e.Type == ai.StreamEventReasoningDelta {
			hasReasoning = true
			if e.TextDelta != "thinking..." {
				t.Errorf("unexpected reasoning delta: %q", e.TextDelta)
			}
		}
	}
	if !hasReasoning {
		t.Error("expected a reasoning delta event")
	}
}

func TestOpenAISSE_FunctionCallDelta(t *testing.T) {
	sse := `data: {"type":"response.output_item.added","item":{"type":"function_call","id":"item_1","call_id":"call_abc","name":"search","arguments":""}}
data: {"type":"response.function_call_arguments.delta","item_id":"item_1","delta":"{\"q\":\""}
data: {"type":"response.function_call_arguments.delta","item_id":"item_1","delta":"hello\"}"}
data: {"type":"response.completed","response":{"id":"resp_3","status":"completed"}}
`
	events := collectEvents(streamFromString(sse))

	toolEvents := 0
	for _, e := range events {
		if e.Type == ai.StreamEventToolCallDelta {
			toolEvents++
			if e.ToolCallID != "call_abc" {
				t.Errorf("expected call_id=call_abc, got %q", e.ToolCallID)
			}
		}
	}
	if toolEvents < 2 {
		t.Errorf("expected at least 2 tool call delta events, got %d", toolEvents)
	}
}

func TestOpenAISSE_SourceEvents(t *testing.T) {
	sse := `data: {"type":"response.web_search_call.sources","sources":[{"type":"url","id":"src_1","url":"https://example.com","title":"Example"}]}
data: {"type":"response.completed","response":{"id":"resp_4","status":"completed"}}
`
	events := collectEvents(streamFromString(sse))

	sourceCount := 0
	for _, e := range events {
		if e.Type == ai.StreamEventSource {
			sourceCount++
			if e.Source == nil {
				t.Error("expected non-nil Source")
				continue
			}
			if e.Source.URL != "https://example.com" {
				t.Errorf("expected URL=https://example.com, got %q", e.Source.URL)
			}
			if e.Source.Title != "Example" {
				t.Errorf("expected Title=Example, got %q", e.Source.Title)
			}
		}
	}
	if sourceCount != 1 {
		t.Errorf("expected 1 source event, got %d", sourceCount)
	}
}

func TestOpenAISSE_ProviderMetadataOnFinish(t *testing.T) {
	sse := `data: {"type":"response.completed","response":{"id":"resp_meta","status":"completed"}}
`
	events := collectEvents(streamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventFinish {
			if e.ProviderMetadata == nil {
				t.Fatal("expected ProviderMetadata on finish event")
			}
			openai, ok := e.ProviderMetadata["openai"].(map[string]any)
			if !ok {
				t.Fatalf("expected openai metadata map, got %T", e.ProviderMetadata["openai"])
			}
			if openai["responseId"] != "resp_meta" {
				t.Errorf("expected responseId=resp_meta, got %v", openai["responseId"])
			}
			return
		}
	}
	t.Error("no finish event found")
}

func TestOpenAISSE_ErrorEvent(t *testing.T) {
	sse := `data: {"type":"error","error":{"code":"rate_limit","message":"Too many requests"}}
`
	events := collectEvents(streamFromString(sse))

	for _, e := range events {
		if e.Type == ai.StreamEventError {
			if e.Error == nil {
				t.Error("expected non-nil error")
			}
			return
		}
	}
	t.Error("expected an error event")
}

func TestOpenAISSE_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	sse := `data: {"type":"response.output_text.delta","delta":"text"}
`
	ch := make(chan ai.StreamEvent, 128)
	decodeResponsesSSEStream(ctx, streamFromString(sse), ch)

	var events []ai.StreamEvent
	for e := range ch {
		events = append(events, e)
	}

	for _, e := range events {
		if e.Type == ai.StreamEventError {
			return
		}
	}
	t.Error("expected an error event on context cancellation")
}

// Request encoder tests

func TestEncodeRequest_SystemAndUserMessage(t *testing.T) {
	req := ai.LanguageModelRequest{
		System:   "You are helpful",
		Messages: []ai.Message{ai.UserMessage("hello")},
	}
	r, _, err := encodeRequest("gpt-4o", req, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.Model != "gpt-4o" {
		t.Errorf("expected model=gpt-4o, got %q", r.Model)
	}
	if !r.Stream {
		t.Error("expected stream=true")
	}
	if len(r.Input) != 2 {
		t.Fatalf("expected 2 input items (system + user), got %d", len(r.Input))
	}
	if r.Input[0].Role != "system" {
		t.Errorf("expected first item role=system, got %q", r.Input[0].Role)
	}
	if r.Input[1].Role != "user" {
		t.Errorf("expected second item role=user, got %q", r.Input[1].Role)
	}
}

func TestEncodeRequest_PreviousResponseID(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("continue")},
		ProviderOptions: map[string]any{
			"openai": ProviderOptions{PreviousResponseID: "resp_abc"},
		},
	}
	r, _, err := encodeRequest("gpt-4o", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.PreviousResponseID != "resp_abc" {
		t.Errorf("expected previous_response_id=resp_abc, got %q", r.PreviousResponseID)
	}
}

func TestEncodeRequest_ReasoningOptions(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("think")},
		ProviderOptions: map[string]any{
			"openai": ProviderOptions{
				ReasoningEffort:  "high",
				ReasoningSummary: "detailed",
			},
		},
	}
	r, _, err := encodeRequest("o3", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.Reasoning == nil {
		t.Fatal("expected reasoning config, got nil")
	}
	if r.Reasoning.Effort != "high" {
		t.Errorf("expected effort=high, got %q", r.Reasoning.Effort)
	}
	if r.Reasoning.Summary != "detailed" {
		t.Errorf("expected summary=detailed, got %q", r.Reasoning.Summary)
	}
}

func TestEncodeRequest_WebSearch(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search something")},
		ProviderOptions: map[string]any{
			"openai": ProviderOptions{EnableWebSearch: true, IncludeSources: true},
		},
	}
	r, _, err := encodeRequest("gpt-4o", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	hasWebSearch := false
	for _, tool := range r.Tools {
		if tool.Type == "web_search_preview" {
			hasWebSearch = true
		}
	}
	if !hasWebSearch {
		t.Error("expected web_search_preview tool")
	}

	hasInclude := false
	for _, inc := range r.Include {
		if inc == "web_search_call.action.sources" {
			hasInclude = true
		}
	}
	if !hasInclude {
		t.Error("expected web_search_call.action.sources in include list")
	}
}

func TestEncodeRequest_FileIDInput(t *testing.T) {
	msg := ai.Message{
		Role: ai.RoleUser,
		Content: []ai.ContentPart{
			ai.TextPart("what is in this file?"),
			{Type: ai.ContentPartTypeFile, FileID: "file-abc123"},
		},
	}
	req := ai.LanguageModelRequest{Messages: []ai.Message{msg}}
	r, _, err := encodeRequest("gpt-4o", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(r.Input) != 1 {
		t.Fatalf("expected 1 input item, got %d", len(r.Input))
	}
	parts := r.Input[0].Content
	if len(parts) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(parts))
	}
	filePart := parts[1]
	if filePart.FileID != "file-abc123" {
		t.Errorf("expected file_id=file-abc123, got %q", filePart.FileID)
	}
}

func TestEncodeRequest_ImageDataInput(t *testing.T) {
	data := []byte("fake-png-bytes")
	msg := ai.Message{
		Role: ai.RoleUser,
		Content: []ai.ContentPart{
			ai.TextPart("describe this image"),
			ai.ImageDataPart(data, "image/png"),
		},
	}
	req := ai.LanguageModelRequest{Messages: []ai.Message{msg}}
	r, _, err := encodeRequest("gpt-4o", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := r.Input[0].Content
	imgPart := parts[1]
	if imgPart.Type != "input_image" {
		t.Errorf("expected type=input_image, got %q", imgPart.Type)
	}
	if imgPart.ImageURL == "" {
		t.Error("expected ImageURL to be set with data URI")
	}
	if imgPart.FileID != "" {
		t.Errorf("expected FileID to be empty, got %q", imgPart.FileID)
	}
}

func TestEncodeRequest_ImageFileIDInput(t *testing.T) {
	msg := ai.Message{
		Role: ai.RoleUser,
		Content: []ai.ContentPart{
			ai.TextPart("describe this image"),
			ai.ImageFileIDPart("file-img123"),
		},
	}
	req := ai.LanguageModelRequest{Messages: []ai.Message{msg}}
	r, _, err := encodeRequest("gpt-4o", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := r.Input[0].Content
	imgPart := parts[1]
	if imgPart.Type != "input_image" {
		t.Errorf("expected type=input_image, got %q", imgPart.Type)
	}
	if imgPart.FileID != "file-img123" {
		t.Errorf("expected FileID=file-img123, got %q", imgPart.FileID)
	}
}

func TestEncodeRequest_FileDataInput(t *testing.T) {
	data := []byte("pdf-content")
	msg := ai.Message{
		Role: ai.RoleUser,
		Content: []ai.ContentPart{
			ai.TextPart("summarize this file"),
			ai.FileDataPart(data, "application/pdf", "doc.pdf"),
		},
	}
	req := ai.LanguageModelRequest{Messages: []ai.Message{msg}}
	r, _, err := encodeRequest("gpt-4o", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := r.Input[0].Content
	filePart := parts[1]
	if filePart.Type != "input_file" {
		t.Errorf("expected type=input_file, got %q", filePart.Type)
	}
	if filePart.FileURL == "" {
		t.Error("expected FileURL to be set with data URI")
	}
	if filePart.Filename != "doc.pdf" {
		t.Errorf("expected Filename=doc.pdf, got %q", filePart.Filename)
	}
}

func TestEncodeRequest_StopSequencesWarning(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
		Settings: ai.CallSettings{StopSequences: []string{"stop"}},
	}
	_, warnings, err := encodeRequest("gpt-4o", req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	found := false
	for _, w := range warnings {
		if w.Type == "unsupported-setting" && w.Setting == "stopSequences" {
			found = true
		}
	}
	if !found {
		t.Error("expected unsupported-setting warning for stopSequences")
	}
}

// Non-stream decoder tests

func TestDecodeNonStream_TextResponse(t *testing.T) {
	body := []byte(`{
		"id": "resp_abc",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"content": [
					{"type": "output_text", "text": "Hello, world!"}
				]
			}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
	}`)

	result, err := decodeResponsesNonStream(body, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "Hello, world!" {
		t.Errorf("expected Text=Hello, world!, got %q", result.Text)
	}
	if result.TotalUsage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", result.TotalUsage.PromptTokens)
	}
	if result.FinishReason != ai.FinishReasonStop {
		t.Errorf("expected FinishReasonStop, got %q", result.FinishReason)
	}
	if result.RawFinishReason != "completed" {
		t.Errorf("expected RawFinishReason=completed, got %q", result.RawFinishReason)
	}
	meta, _ := result.ProviderMetadata["openai"].(map[string]any)
	if meta["responseId"] != "resp_abc" {
		t.Errorf("expected responseId=resp_abc, got %v", meta["responseId"])
	}
}

func TestDecodeNonStream_ErrorResponse(t *testing.T) {
	body := []byte(`{"error":{"code":"invalid_request","message":"bad request"}}`)
	_, err := decodeResponsesNonStream(body, nil)
	if err == nil {
		t.Error("expected error, got nil")
	}
}
