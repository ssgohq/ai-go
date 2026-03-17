package gemini

import (
	"encoding/json"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

// testSearchModel is a model ID used in Google Search grounding tests.
const testSearchModel = "gemini-2.5-flash"

// encodeFullRequest simulates the complete request encoding pipeline including
// extra body fields and TransformRequestBody, returning the final JSON body as a map.
func encodeFullRequest(req ai.LanguageModelRequest) (map[string]any, error) {
	return encodeFullRequestForModel(testSearchModel, req)
}

func encodeFullRequestForModel(modelID string, req ai.LanguageModelRequest) (map[string]any, error) {
	cr, err := openaichat.EncodeRequest(openaichat.EncodeRequestParams{
		ModelID:            modelID,
		SanitizeTools:      sanitizeToolSchemas,
		IncludeStreamUsage: true,
	}, req, true)
	if err != nil {
		return nil, err
	}

	// Marshal to map (same as openaichat.LanguageModel.Stream does).
	raw, err := json.Marshal(cr)
	if err != nil {
		return nil, err
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}

	// Merge extra body fields.
	extraFields := extraBodyFieldsForRequest(req)
	for k, v := range extraFields {
		body[k] = v
	}

	// Apply transform.
	body = mergeGoogleSearchTools(body)

	return body, nil
}

// toolsFromBody extracts the tools array from an encoded body map.
func toolsFromBody(body map[string]any) []map[string]any {
	toolsRaw, ok := body["tools"]
	if !ok {
		return nil
	}
	switch tools := toolsRaw.(type) {
	case []any:
		result := make([]map[string]any, 0, len(tools))
		for _, t := range tools {
			if m, ok := t.(map[string]any); ok {
				result = append(result, m)
			}
		}
		return result
	case []map[string]any:
		return tools
	default:
		return nil
	}
}

func TestGoogleSearch_EnabledWithNoOtherTools(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("What is the latest news?")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	body, err := encodeFullRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := toolsFromBody(body)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool (google_search), got %d", len(tools))
	}
	if _, ok := tools[0]["googleSearch"]; !ok {
		t.Errorf("expected tool to have googleSearch key, got %v", tools[0])
	}
	// Must not have the old OpenAI-style "type" key.
	if _, ok := tools[0]["type"]; ok {
		t.Error("googleSearch tool must not have a type key")
	}
	// Must not have a "function" key.
	if _, ok := tools[0]["function"]; ok {
		t.Error("googleSearch tool must not have a function key")
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

	body, err := encodeFullRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := toolsFromBody(body)
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools (function + google_search), got %d", len(tools))
	}
	// First tool is the function.
	if tools[0]["type"] != "function" {
		t.Errorf("expected first tool type=function, got %v", tools[0]["type"])
	}
	// Last tool is googleSearch (native format).
	last := tools[len(tools)-1]
	if _, ok := last["googleSearch"]; !ok {
		t.Errorf("expected last tool to have googleSearch key, got %v", last)
	}
}

func TestGoogleSearch_DisabledByDefault(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hello")},
	}

	body, err := encodeFullRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := toolsFromBody(body)
	if len(tools) != 0 {
		t.Errorf("expected no tools when google search not enabled, got %d", len(tools))
	}
}

func TestGoogleSearch_ExplicitlyDisabled(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hello")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: false},
		},
	}

	body, err := encodeFullRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := toolsFromBody(body)
	if len(tools) != 0 {
		t.Errorf("expected no tools when EnableGoogleSearch=false, got %d", len(tools))
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

	body, err := encodeFullRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := toolsFromBody(body)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	tool := tools[0]
	searchCfg, ok := tool["googleSearch"].(map[string]any)
	if !ok {
		t.Fatalf("expected googleSearch config map, got %T", tool["googleSearch"])
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

	body, err := encodeFullRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := toolsFromBody(body)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	tool := tools[0]
	searchCfg, ok := tool["googleSearch"].(map[string]any)
	if !ok {
		t.Fatalf("expected googleSearch config map, got %T", tool["googleSearch"])
	}

	typesRaw, ok := searchCfg["search_types"]
	if !ok {
		t.Fatal("expected search_types key in google_search config")
	}
	// Extra body fields are merged after JSON round-trip, so the slice
	// retains its original Go type ([]string).
	switch types := typesRaw.(type) {
	case []string:
		if len(types) != 2 || types[0] != "web" || types[1] != "image" {
			t.Errorf("expected search_types=[web image], got %v", types)
		}
	case []any:
		if len(types) != 2 || types[0] != "web" || types[1] != "image" {
			t.Errorf("expected search_types=[web image], got %v", types)
		}
	default:
		t.Fatalf("expected search_types []string or []any, got %T", typesRaw)
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

	body, err := encodeFullRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := toolsFromBody(body)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	tool := tools[0]
	searchCfg, ok := tool["googleSearch"].(map[string]any)
	if !ok {
		t.Fatalf("expected googleSearch config map, got %T", tool["googleSearch"])
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
	// When GoogleSearchConfig is nil, the tool should be {"googleSearch": {}}.
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	body, err := encodeFullRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := toolsFromBody(body)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	searchCfg, ok := tools[0]["googleSearch"].(map[string]any)
	if !ok {
		t.Fatalf("expected googleSearch key with map value, got %T", tools[0]["googleSearch"])
	}
	if len(searchCfg) != 0 {
		t.Errorf("expected empty googleSearch config when GoogleSearchConfig is nil, got %v", searchCfg)
	}
}

// --- Model tests ---

func TestGoogleSearch_AllModels_ToolAdded(t *testing.T) {
	models := []string{
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-2.5-pro",
		"gemini-3-flash-preview",
		"gemini-3-pro-preview",
	}

	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("latest news")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	for _, modelID := range models {
		t.Run(modelID, func(t *testing.T) {
			body, err := encodeFullRequestForModel(modelID, req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tools := toolsFromBody(body)
			if len(tools) != 1 {
				t.Errorf("model %s: expected 1 google_search tool, got %d", modelID, len(tools))
			}
			if len(tools) > 0 {
				if _, ok := tools[0]["googleSearch"]; !ok {
					t.Errorf("model %s: expected tool with googleSearch key, got %v", modelID, tools[0])
				}
			}
		})
	}
}

func TestGoogleSearch_ProModel_NoWarning(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("search")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{EnableGoogleSearch: true},
		},
	}

	warnings := warningsForRequest("gemini-2.5-pro", req)
	for _, w := range warnings {
		if w.Setting == "enableGoogleSearch" {
			t.Errorf("unexpected unsupported-model warning for gemini-2.5-pro")
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

	warnings := warningsForRequest(testSearchModel, req)
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

	warnings := warningsForRequest(testSearchModel, req)
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

	warnings := warningsForRequest(testSearchModel, req)
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

	warnings := warningsForRequest(testSearchModel, req)
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

	warnings := warningsForRequest(testSearchModel, req)
	if len(warnings) != 0 {
		t.Errorf("expected no warnings for normal search request, got %d", len(warnings))
	}
}
