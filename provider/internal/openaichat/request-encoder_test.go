package openaichat_test

import (
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
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
			InputSchema: map[string]any{"type": "object"},
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

// --- ToolChoice encoding contract ---

func TestEncodeRequest_ToolChoice_Default_Auto(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
		Tools: []ai.ToolDefinition{{
			Name:        "search",
			InputSchema: map[string]any{"type": "object"},
		}},
		ToolChoice: nil, // nil → "auto"
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.ToolChoice != "auto" {
		t.Errorf("expected auto, got %v", cr.ToolChoice)
	}
}

func TestEncodeRequest_ToolChoice_None(t *testing.T) {
	tc := ai.ToolChoiceNone
	req := ai.LanguageModelRequest{
		Messages:   []ai.Message{ai.UserMessage("hi")},
		Tools:      []ai.ToolDefinition{{Name: "t", InputSchema: map[string]any{"type": "object"}}},
		ToolChoice: &tc,
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.ToolChoice != "none" {
		t.Errorf("expected none, got %v", cr.ToolChoice)
	}
}

func TestEncodeRequest_ToolChoice_Required(t *testing.T) {
	tc := ai.ToolChoiceRequired
	req := ai.LanguageModelRequest{
		Messages:   []ai.Message{ai.UserMessage("hi")},
		Tools:      []ai.ToolDefinition{{Name: "t", InputSchema: map[string]any{"type": "object"}}},
		ToolChoice: &tc,
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.ToolChoice != "required" {
		t.Errorf("expected required, got %v", cr.ToolChoice)
	}
}

func TestEncodeRequest_ToolChoice_Specific(t *testing.T) {
	tc := ai.ToolChoiceSpecific("my_tool")
	req := ai.LanguageModelRequest{
		Messages:   []ai.Message{ai.UserMessage("hi")},
		Tools:      []ai.ToolDefinition{{Name: "my_tool", InputSchema: map[string]any{"type": "object"}}},
		ToolChoice: &tc,
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, ok := cr.ToolChoice.(map[string]any)
	if !ok {
		t.Fatalf("expected map for specific tool choice, got %T", cr.ToolChoice)
	}
	if obj["type"] != "function" {
		t.Errorf("expected type=function, got %v", obj["type"])
	}
	fn, _ := obj["function"].(map[string]any)
	if fn["name"] != "my_tool" {
		t.Errorf("expected name=my_tool, got %v", fn["name"])
	}
}

// --- Output encoding ---

func TestEncodeRequest_Output_JSONObject(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("return json")},
		Output:   ai.OutputJSONObject(),
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.ResponseFormat == nil {
		t.Fatal("expected ResponseFormat to be set")
	}
	if cr.ResponseFormat.Type != "json_object" {
		t.Errorf("expected json_object, got %s", cr.ResponseFormat.Type)
	}
	if cr.ResponseFormat.JSONSchema != nil {
		t.Error("expected no JSONSchema for json_object mode")
	}
}

func TestEncodeRequest_Output_JSONSchema(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("return structured")},
		Output: ai.OutputObject(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}),
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.ResponseFormat == nil {
		t.Fatal("expected ResponseFormat")
	}
	if cr.ResponseFormat.Type != "json_schema" {
		t.Errorf("expected json_schema, got %s", cr.ResponseFormat.Type)
	}
	if cr.ResponseFormat.JSONSchema == nil {
		t.Error("expected JSONSchema to be set for object mode")
	}
}

func TestEncodeRequest_Output_Text_NoResponseFormat(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("tell me a story")},
		Output:   ai.OutputText(),
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.ResponseFormat != nil {
		t.Error("expected no ResponseFormat for text output")
	}
}

// --- CallSettings: TopP and Seed propagation ---

func TestEncodeRequest_Settings_TopP(t *testing.T) {
	topP := float32(0.95)
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
		Settings: ai.CallSettings{TopP: &topP},
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.TopP != 0.95 {
		t.Errorf("expected TopP=0.95, got %v", cr.TopP)
	}
}

func TestEncodeRequest_Settings_Seed(t *testing.T) {
	seed := 42
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
		Settings: ai.CallSettings{Seed: &seed},
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.Seed == nil || *cr.Seed != 42 {
		t.Errorf("expected Seed=42, got %v", cr.Seed)
	}
}

func TestEncodeRequest_Settings_NilTopP_NotIncluded(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hi")},
		Settings: ai.CallSettings{}, // TopP nil
	}
	cr, err := openaichat.EncodeRequest(defaultEncodeParams("m"), req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cr.TopP != 0 {
		t.Errorf("expected TopP=0 (zero value) when not set, got %v", cr.TopP)
	}
}
