package gemini

import (
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

// supportedSearchModel is a model ID that supports Google Search grounding.
const supportedSearchModel = "gemini-2.5-flash"

// unsupportedSearchModel is a model ID that does NOT support Google Search grounding.
const unsupportedSearchModel = "gemini-2.5-pro"

// encodeWithExtraTools is a test helper that runs EncodeRequest with ExtraTools derived
// from gemini ProviderOptions so tests don't need to spin up an HTTP server.
// Uses a supported model by default so Google Search is not filtered out.
func encodeWithExtraTools(req ai.LanguageModelRequest) (openaichat.ChatRequest, error) {
	return encodeWithExtraToolsForModel(supportedSearchModel, req)
}

func encodeWithExtraToolsForModel(modelID string, req ai.LanguageModelRequest) (openaichat.ChatRequest, error) {
	extra := extraToolsForRequest(modelID, req)
	return openaichat.EncodeRequest(openaichat.EncodeRequestParams{
		ModelID:            modelID,
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

func TestGoogleSearch_WithDynamicRetrievalThreshold(t *testing.T) {
	threshold := 0.7
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("news today")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{
				EnableGoogleSearch: true,
				GoogleSearchConfig: &GoogleSearchConfig{
					DynamicRetrievalThreshold: &threshold,
				},
			},
		},
	}

	cr, err := encodeWithExtraTools(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cr.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(cr.Tools))
	}

	tool := cr.Tools[0]
	if tool["type"] != "google_search" {
		t.Errorf("expected type=google_search, got %v", tool["type"])
	}

	searchCfg, ok := tool["google_search"].(map[string]any)
	if !ok {
		t.Fatalf("expected google_search config map, got %T", tool["google_search"])
	}

	dynCfg, ok := searchCfg["dynamic_retrieval_config"].(map[string]any)
	if !ok {
		t.Fatalf("expected dynamic_retrieval_config map, got %T", searchCfg["dynamic_retrieval_config"])
	}
	if dynCfg["mode"] != "MODE_DYNAMIC" {
		t.Errorf("expected mode=MODE_DYNAMIC, got %v", dynCfg["mode"])
	}
	if dynCfg["dynamic_threshold"] != threshold {
		t.Errorf("expected dynamic_threshold=%v, got %v", threshold, dynCfg["dynamic_threshold"])
	}
}

func TestGoogleSearch_WithSearchTypes(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("find images")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{
				EnableGoogleSearch: true,
				GoogleSearchConfig: &GoogleSearchConfig{
					SearchTypes: []string{"web", "image"},
				},
			},
		},
	}

	cr, err := encodeWithExtraTools(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cr.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(cr.Tools))
	}

	tool := cr.Tools[0]
	searchCfg, ok := tool["google_search"].(map[string]any)
	if !ok {
		t.Fatalf("expected google_search config map, got %T", tool["google_search"])
	}

	types, ok := searchCfg["search_types"].([]string)
	if !ok {
		t.Fatalf("expected search_types []string, got %T", searchCfg["search_types"])
	}
	if len(types) != 2 || types[0] != "web" || types[1] != "image" {
		t.Errorf("expected search_types=[web image], got %v", types)
	}
}

func TestGoogleSearch_WithTimeRangeFilter(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("recent events")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{
				EnableGoogleSearch: true,
				GoogleSearchConfig: &GoogleSearchConfig{
					TimeRangeFilter: &TimeRangeFilter{
						StartTime: "2024-01-01T00:00:00Z",
						EndTime:   "2024-12-31T23:59:59Z",
					},
				},
			},
		},
	}

	cr, err := encodeWithExtraTools(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cr.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(cr.Tools))
	}

	tool := cr.Tools[0]
	searchCfg, ok := tool["google_search"].(map[string]any)
	if !ok {
		t.Fatalf("expected google_search config map, got %T", tool["google_search"])
	}

	trf, ok := searchCfg["time_range_filter"].(map[string]any)
	if !ok {
		t.Fatalf("expected time_range_filter map, got %T", searchCfg["time_range_filter"])
	}
	if trf["start_time"] != "2024-01-01T00:00:00Z" {
		t.Errorf("expected start_time=2024-01-01T00:00:00Z, got %v", trf["start_time"])
	}
	if trf["end_time"] != "2024-12-31T23:59:59Z" {
		t.Errorf("expected end_time=2024-12-31T23:59:59Z, got %v", trf["end_time"])
	}
}

func TestGoogleSearch_NoConfigNoGoogleSearchKey(t *testing.T) {
	// When GoogleSearchConfig is nil, the tool should have no "google_search" key.
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	cr, err := encodeWithExtraTools(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cr.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(cr.Tools))
	}
	if _, ok := cr.Tools[0]["google_search"]; ok {
		t.Error("expected no google_search config key when GoogleSearchConfig is nil")
	}
}

// --- Model restriction tests (bead aisdk-27j.4) ---

func TestGoogleSearch_UnsupportedModel_ToolNotAdded(t *testing.T) {
	// gemini-2.5-pro does NOT support google_search; tool should be omitted.
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("latest news")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	cr, err := encodeWithExtraToolsForModel(unsupportedSearchModel, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cr.Tools) != 0 {
		t.Errorf("expected no tools for unsupported model %s, got %d", unsupportedSearchModel, len(cr.Tools))
	}
}

func TestGoogleSearch_SupportedModels_ToolAdded(t *testing.T) {
	supported := []string{
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-3-flash-preview",
		"gemini-3-pro-preview",
	}

	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("latest news")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	for _, modelID := range supported {
		t.Run(modelID, func(t *testing.T) {
			cr, err := encodeWithExtraToolsForModel(modelID, req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(cr.Tools) != 1 {
				t.Errorf("model %s: expected 1 google_search tool, got %d", modelID, len(cr.Tools))
			}
			if len(cr.Tools) > 0 && cr.Tools[0]["type"] != "google_search" {
				t.Errorf("model %s: expected type=google_search, got %v", modelID, cr.Tools[0]["type"])
			}
		})
	}
}

func TestGoogleSearch_UnsupportedModel_WarningEmitted(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	warnings := warningsForRequest(unsupportedSearchModel, req)
	if len(warnings) == 0 {
		t.Fatal("expected warning for unsupported model, got none")
	}

	found := false
	for _, w := range warnings {
		if w.Setting == "enableGoogleSearch" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected enableGoogleSearch warning, got warnings: %v", warnings)
	}
}

func TestGoogleSearch_SupportedModel_NoUnsupportedWarning(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	warnings := warningsForRequest(supportedSearchModel, req)
	for _, w := range warnings {
		if w.Setting == "enableGoogleSearch" {
			t.Errorf("unexpected unsupported-model warning for supported model %s", supportedSearchModel)
		}
	}
}

// --- Warning tests ---

func TestGoogleSearch_WarningTopKWithSearch(t *testing.T) {
	topK := 40
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		Settings: ai.CallSettings{TopK: &topK},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	warnings := warningsForRequest(supportedSearchModel, req)
	if len(warnings) != 1 {
		t.Fatalf("expected 1 warning, got %d: %v", len(warnings), warnings)
	}
	if warnings[0].Type != "unsupported-setting" {
		t.Errorf("expected type=unsupported-setting, got %v", warnings[0].Type)
	}
	if warnings[0].Setting != "topK" {
		t.Errorf("expected setting=topK, got %v", warnings[0].Setting)
	}
}

func TestGoogleSearch_WarningSeedWithSearch(t *testing.T) {
	seed := 42
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		Settings: ai.CallSettings{Seed: &seed},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	warnings := warningsForRequest(supportedSearchModel, req)
	if len(warnings) != 1 {
		t.Fatalf("expected 1 warning, got %d: %v", len(warnings), warnings)
	}
	if warnings[0].Type != "unsupported-setting" {
		t.Errorf("expected type=unsupported-setting, got %v", warnings[0].Type)
	}
	if warnings[0].Setting != "seed" {
		t.Errorf("expected setting=seed, got %v", warnings[0].Setting)
	}
}

func TestGoogleSearch_WarningBothTopKAndSeedWithSearch(t *testing.T) {
	topK := 40
	seed := 42
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		Settings: ai.CallSettings{TopK: &topK, Seed: &seed},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	warnings := warningsForRequest(supportedSearchModel, req)
	if len(warnings) != 2 {
		t.Fatalf("expected 2 warnings, got %d: %v", len(warnings), warnings)
	}
}

func TestGoogleSearch_NoWarningsWithoutSearch(t *testing.T) {
	topK := 40
	seed := 42
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hello")},
		Settings: ai.CallSettings{TopK: &topK, Seed: &seed},
	}

	warnings := warningsForRequest(supportedSearchModel, req)
	if len(warnings) != 0 {
		t.Errorf("expected no warnings when search is disabled, got %d", len(warnings))
	}
}

func TestGoogleSearch_NoWarningsNormalSearch(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	warnings := warningsForRequest(supportedSearchModel, req)
	if len(warnings) != 0 {
		t.Errorf("expected no warnings for normal search request, got %d", len(warnings))
	}
}
