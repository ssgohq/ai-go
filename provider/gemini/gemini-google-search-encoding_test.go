package gemini

import (
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

// encodeWithExtraTools is a test helper that runs EncodeRequest with ExtraTools derived
// from gemini ProviderOptions so tests don't need to spin up an HTTP server.
func encodeWithExtraTools(req ai.LanguageModelRequest) (openaichat.ChatRequest, error) {
	extra := extraToolsForRequest(req)
	return openaichat.EncodeRequest(openaichat.EncodeRequestParams{
		ModelID:            "gemini-2.5-pro",
		SanitizeTools:      sanitizeToolSchemas,
		IncludeStreamUsage: true,
		ExtraTools:         extra,
	}, req, true)
}

func TestGoogleSearch_EnabledWithNoOtherTools(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("What is the latest news?")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	cr, err := encodeWithExtraTools(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(cr.Tools) != 1 {
		t.Fatalf("expected 1 tool (google_search), got %d", len(cr.Tools))
	}
	if cr.Tools[0]["type"] != "google_search" {
		t.Errorf("expected type=google_search, got %v", cr.Tools[0]["type"])
	}
	// google_search has no "function" key
	if _, ok := cr.Tools[0]["function"]; ok {
		t.Error("google_search tool must not have a function key")
	}
}

func TestGoogleSearch_EnabledWithFunctionTools(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search and call my tool")},
		Tools: []ai.ToolDefinition{
			{Name: "my_tool", Description: "does something", InputSchema: map[string]any{"type": "object"}},
		},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	cr, err := encodeWithExtraTools(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(cr.Tools) != 2 {
		t.Fatalf("expected 2 tools (function + google_search), got %d", len(cr.Tools))
	}
	// First tool is the function
	if cr.Tools[0]["type"] != "function" {
		t.Errorf("expected first tool type=function, got %v", cr.Tools[0]["type"])
	}
	// Last tool is google_search
	last := cr.Tools[len(cr.Tools)-1]
	if last["type"] != "google_search" {
		t.Errorf("expected last tool type=google_search, got %v", last["type"])
	}
}

func TestGoogleSearch_DisabledByDefault(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hello")},
	}

	cr, err := encodeWithExtraTools(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(cr.Tools) != 0 {
		t.Errorf("expected no tools when google search not enabled, got %d", len(cr.Tools))
	}
}

func TestGoogleSearch_ExplicitlyDisabled(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hello")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: false},
		},
	}

	cr, err := encodeWithExtraTools(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(cr.Tools) != 0 {
		t.Errorf("expected no tools when EnableGoogleSearch=false, got %d", len(cr.Tools))
	}
}

func TestGoogleSearch_ParseProviderOptions_WrongType(t *testing.T) {
	opts := parseProviderOptions(map[string]any{
		"gemini": "not-a-struct",
	})
	if opts.EnableGoogleSearch {
		t.Error("expected EnableGoogleSearch=false for wrong type")
	}
}

func TestGoogleSearch_ParseProviderOptions_NilMap(t *testing.T) {
	opts := parseProviderOptions(nil)
	if opts.EnableGoogleSearch {
		t.Error("expected EnableGoogleSearch=false for nil map")
	}
}

func TestGoogleSearch_ParseProviderOptions_MissingKey(t *testing.T) {
	opts := parseProviderOptions(map[string]any{"other": "value"})
	if opts.EnableGoogleSearch {
		t.Error("expected EnableGoogleSearch=false when gemini key missing")
	}
}
