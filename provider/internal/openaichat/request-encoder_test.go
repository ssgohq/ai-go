package openaichat_test

import (
	"testing"

	"github.com/ssgohq/ai-go/ai"
	"github.com/ssgohq/ai-go/provider/internal/openaichat"
)

func defaultEncodeParams(modelID string) openaichat.EncodeRequestParams {
	return openaichat.EncodeRequestParams{
		ModelID:            modelID,
		IncludeStreamUsage: true,
	}
}

func TestEncodeRequest_SystemAndMessages(t *testing.T) {
	req := ai.LanguageModelRequest{
		System:   "You are helpful",
		Messages: []ai.Message{ai.UserMessage("hi")},
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("test-model"), req, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.Model != "test-model" {
		t.Errorf("unexpected model: %s", cr.Model)
	}
	if !cr.Stream {
		t.Error("expected streaming to be true")
	}
	if len(cr.Messages) != 2 {
		t.Errorf("expected 2 messages (system + user), got %d", len(cr.Messages))
	}
	if cr.Messages[0]["role"] != "system" {
		t.Error("first message should be system")
	}
}

func TestEncodeRequest_StreamOptions(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("test-model"), req, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.StreamOptions == nil {
		t.Fatal("expected stream_options to be set for streaming request")
	}
	if cr.StreamOptions["include_usage"] != true {
		t.Error("expected include_usage=true in stream_options")
	}
}

func TestEncodeRequest_NoStreamOptions_WhenDisabled(t *testing.T) {
	params := openaichat.EncodeRequestParams{
		ModelID:            "test-model",
		IncludeStreamUsage: false,
	}
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	}
	cr, err := openaichat.EncodeRequest(params, req, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.StreamOptions != nil {
		t.Error("expected stream_options to be nil when IncludeStreamUsage=false")
	}
}

func TestEncodeRequest_MultimodalImageURL(t *testing.T) {
	msg := ai.Message{
		Role: ai.RoleUser,
		Content: []ai.ContentPart{
			ai.TextPart("what is this?"),
			ai.ImageURLPart("https://example.com/img.png"),
		},
	}
	req := ai.LanguageModelRequest{Messages: []ai.Message{msg}}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("test-model"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	content, ok := cr.Messages[0]["content"].([]map[string]any)
	if !ok || len(content) != 2 {
		t.Errorf("expected multipart content with 2 parts, got %v", cr.Messages[0]["content"])
	}
}

func TestEncodeRequest_SanitizeToolsHook(t *testing.T) {
	sanitizeCalled := false
	params := openaichat.EncodeRequestParams{
		ModelID: "test-model",
		SanitizeTools: func(tools []map[string]any) []map[string]any {
			sanitizeCalled = true
			return tools
		},
	}
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
		Tools: []ai.ToolDefinition{{
			Name:        "search",
			Description: "web search",
			Parameters:  map[string]any{"type": "object"},
		}},
	}
	_, err := openaichat.EncodeRequest(params, req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !sanitizeCalled {
		t.Error("expected SanitizeTools hook to be called")
	}
}

func TestEncodeRequest_NonStreamHasNoStreamOptions(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("test-model"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.Stream {
		t.Error("expected stream=false for non-streaming request")
	}
	if cr.StreamOptions != nil {
		t.Error("expected no stream_options for non-streaming request")
	}
}
