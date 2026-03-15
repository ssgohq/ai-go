package ai_test

import (
	"context"
	"testing"

	"github.com/ssgohq/ai-go/ai"
)

// stubLanguageModel is a minimal ai.LanguageModel for contract regression tests.
type stubLanguageModel struct {
	events []ai.StreamEvent
}

func (m *stubLanguageModel) ModelID() string { return "stub" }

func (m *stubLanguageModel) Stream(_ context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	ch := make(chan ai.StreamEvent, len(m.events)+1)
	for _, ev := range m.events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

// TestProviderContract_ProviderOptionsPassthrough verifies that ProviderOptions
// set on GenerateTextRequest reach the LanguageModel.Stream call.
func TestProviderContract_ProviderOptionsPassthrough(t *testing.T) {
	var capturedOpts map[string]any

	model := &captureOptsModel{capture: &capturedOpts}
	_, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:  model,
		System: "test",
		Messages: []ai.Message{
			ai.UserMessage("hello"),
		},
		ProviderOptions: map[string]any{
			"openai": map[string]any{
				"previousResponseId": "resp_abc123",
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if capturedOpts == nil {
		t.Fatal("ProviderOptions not passed to model")
	}
	openaiOpts, ok := capturedOpts["openai"].(map[string]any)
	if !ok {
		t.Fatalf("expected openai options map, got %T", capturedOpts["openai"])
	}
	if openaiOpts["previousResponseId"] != "resp_abc123" {
		t.Errorf("expected previousResponseId=resp_abc123, got %v", openaiOpts["previousResponseId"])
	}
}

type captureOptsModel struct {
	capture *map[string]any
}

func (m *captureOptsModel) ModelID() string { return "capture-opts" }

func (m *captureOptsModel) Stream(_ context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	*m.capture = req.ProviderOptions
	ch := make(chan ai.StreamEvent, 2)
	ch <- ai.StreamEvent{Type: ai.StreamEventTextDelta, TextDelta: "hello"}
	ch <- ai.StreamEvent{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop, RawFinishReason: "stop"}
	close(ch)
	return ch, nil
}

// TestProviderContract_WarningsPassthrough verifies that warnings emitted by a provider
// are surfaced in the GenerateTextResult.
func TestProviderContract_WarningsPassthrough(t *testing.T) {
	model := &stubLanguageModel{
		events: []ai.StreamEvent{
			{Type: ai.StreamEventTextDelta, TextDelta: "result"},
			{
				Type:         ai.StreamEventFinish,
				FinishReason: ai.FinishReasonStop,
				RawFinishReason: "stop",
				Warnings: []ai.Warning{
					{Type: "unsupported-setting", Setting: "topP", Message: "topP is not supported by this model"},
				},
			},
		},
	}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Warnings) != 1 {
		t.Fatalf("expected 1 warning, got %d", len(result.Warnings))
	}
	w := result.Warnings[0]
	if w.Type != "unsupported-setting" {
		t.Errorf("expected warning type unsupported-setting, got %q", w.Type)
	}
	if w.Setting != "topP" {
		t.Errorf("expected warning setting topP, got %q", w.Setting)
	}
}

// TestProviderContract_RawFinishReason verifies raw finish reason is surfaced.
func TestProviderContract_RawFinishReason(t *testing.T) {
	model := &stubLanguageModel{
		events: []ai.StreamEvent{
			{Type: ai.StreamEventTextDelta, TextDelta: "ok"},
			{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop, RawFinishReason: "end_turn"},
		},
	}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.RawFinishReason != "end_turn" {
		t.Errorf("expected RawFinishReason=end_turn, got %q", result.RawFinishReason)
	}
}

// TestProviderContract_ProviderMetadataPassthrough verifies provider metadata in finish event
// is surfaced in results.
func TestProviderContract_ProviderMetadataPassthrough(t *testing.T) {
	model := &stubLanguageModel{
		events: []ai.StreamEvent{
			{Type: ai.StreamEventTextDelta, TextDelta: "text"},
			{
				Type:         ai.StreamEventFinish,
				FinishReason: ai.FinishReasonStop,
				ProviderMetadata: map[string]any{
					"openai": map[string]any{"responseId": "resp_xyz"},
				},
			},
		},
	}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ProviderMetadata == nil {
		t.Fatal("expected ProviderMetadata, got nil")
	}
	openaiMeta, ok := result.ProviderMetadata["openai"].(map[string]any)
	if !ok {
		t.Fatalf("expected openai metadata, got %T", result.ProviderMetadata["openai"])
	}
	if openaiMeta["responseId"] != "resp_xyz" {
		t.Errorf("expected responseId=resp_xyz, got %v", openaiMeta["responseId"])
	}
}

// TestProviderContract_SourceEvents verifies that source events are collected in the result.
func TestProviderContract_SourceEvents(t *testing.T) {
	model := &stubLanguageModel{
		events: []ai.StreamEvent{
			{Type: ai.StreamEventTextDelta, TextDelta: "search result"},
			{
				Type: ai.StreamEventSource,
				Source: &ai.Source{
					SourceType: "url",
					ID:         "src_1",
					URL:        "https://example.com/article",
					Title:      "Example Article",
				},
			},
			{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop},
		},
	}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("search something")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Sources) != 1 {
		t.Fatalf("expected 1 source, got %d", len(result.Sources))
	}
	src := result.Sources[0]
	if src.URL != "https://example.com/article" {
		t.Errorf("expected URL=https://example.com/article, got %q", src.URL)
	}
	if src.Title != "Example Article" {
		t.Errorf("expected Title=Example Article, got %q", src.Title)
	}
}
