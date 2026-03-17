package gemini

import (
	"encoding/json"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// jsonRoundTrip marshals v to JSON and unmarshals back into a map for comparison.
func jsonRoundTrip(t *testing.T, v any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	return m
}

func TestEncodeNativeTools_NoToolsNoGrounding(t *testing.T) {
	result := encodeNativeTools(nil, nil, ProviderOptions{})
	if len(result.Tools) != 0 {
		t.Errorf("expected no tools, got %d", len(result.Tools))
	}
	if result.ToolConfig != nil {
		t.Errorf("expected no toolConfig, got %v", result.ToolConfig)
	}
}

func TestEncodeNativeTools_SingleFunctionTool(t *testing.T) {
	tools := []ai.ToolDefinition{
		{
			Name:        "get_weather",
			Description: "Get current weather",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		},
	}

	result := encodeNativeTools(tools, nil, ProviderOptions{})

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tools entry, got %d", len(result.Tools))
	}

	m := jsonRoundTrip(t, result)
	toolsArr := m["tools"].([]any)
	funcEntry := toolsArr[0].(map[string]any)
	decls := funcEntry["functionDeclarations"].([]any)
	if len(decls) != 1 {
		t.Fatalf("expected 1 function declaration, got %d", len(decls))
	}

	decl := decls[0].(map[string]any)
	if decl["name"] != "get_weather" {
		t.Errorf("expected name=get_weather, got %v", decl["name"])
	}
	if decl["description"] != "Get current weather" {
		t.Errorf("expected description='Get current weather', got %v", decl["description"])
	}
	params := decl["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("expected parameters.type=object, got %v", params["type"])
	}
}

func TestEncodeNativeTools_MultipleFunctionTools(t *testing.T) {
	tools := []ai.ToolDefinition{
		{Name: "tool_a", Description: "Tool A", InputSchema: map[string]any{"type": "object"}},
		{Name: "tool_b", Description: "Tool B", InputSchema: map[string]any{"type": "object"}},
		{Name: "tool_c", Description: "Tool C", InputSchema: map[string]any{"type": "object"}},
	}

	result := encodeNativeTools(tools, nil, ProviderOptions{})

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tools entry (all grouped), got %d", len(result.Tools))
	}

	m := jsonRoundTrip(t, result)
	toolsArr := m["tools"].([]any)
	funcEntry := toolsArr[0].(map[string]any)
	decls := funcEntry["functionDeclarations"].([]any)
	if len(decls) != 3 {
		t.Fatalf("expected 3 function declarations, got %d", len(decls))
	}

	names := []string{"tool_a", "tool_b", "tool_c"}
	for i, name := range names {
		decl := decls[i].(map[string]any)
		if decl["name"] != name {
			t.Errorf("declaration[%d]: expected name=%s, got %v", i, name, decl["name"])
		}
	}
}

func TestEncodeNativeTools_ToolChoiceAuto(t *testing.T) {
	tc := ai.ToolChoiceAuto
	result := encodeNativeTools(nil, &tc, ProviderOptions{})

	m := jsonRoundTrip(t, result)
	toolConfig := m["toolConfig"].(map[string]any)
	fcc := toolConfig["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "AUTO" {
		t.Errorf("expected mode=AUTO, got %v", fcc["mode"])
	}
}

func TestEncodeNativeTools_ToolChoiceNone(t *testing.T) {
	tc := ai.ToolChoiceNone
	result := encodeNativeTools(nil, &tc, ProviderOptions{})

	m := jsonRoundTrip(t, result)
	toolConfig := m["toolConfig"].(map[string]any)
	fcc := toolConfig["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "NONE" {
		t.Errorf("expected mode=NONE, got %v", fcc["mode"])
	}
}

func TestEncodeNativeTools_ToolChoiceRequired(t *testing.T) {
	tc := ai.ToolChoiceRequired
	result := encodeNativeTools(nil, &tc, ProviderOptions{})

	m := jsonRoundTrip(t, result)
	toolConfig := m["toolConfig"].(map[string]any)
	fcc := toolConfig["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("expected mode=ANY, got %v", fcc["mode"])
	}
	if _, ok := fcc["allowedFunctionNames"]; ok {
		t.Error("expected no allowedFunctionNames for required choice")
	}
}

func TestEncodeNativeTools_ToolChoiceSpecificTool(t *testing.T) {
	tc := ai.ToolChoiceSpecific("my_tool")
	result := encodeNativeTools(nil, &tc, ProviderOptions{})

	m := jsonRoundTrip(t, result)
	toolConfig := m["toolConfig"].(map[string]any)
	fcc := toolConfig["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("expected mode=ANY, got %v", fcc["mode"])
	}
	allowed := fcc["allowedFunctionNames"].([]any)
	if len(allowed) != 1 || allowed[0] != "my_tool" {
		t.Errorf("expected allowedFunctionNames=[my_tool], got %v", allowed)
	}
}

func TestEncodeNativeTools_NilToolChoice(t *testing.T) {
	result := encodeNativeTools(
		[]ai.ToolDefinition{{Name: "t", Description: "d", InputSchema: map[string]any{"type": "object"}}},
		nil,
		ProviderOptions{},
	)

	if result.ToolConfig != nil {
		t.Errorf("expected nil toolConfig for nil tool choice, got %v", result.ToolConfig)
	}
}

func TestEncodeNativeTools_GoogleSearchEnabledNoConfig(t *testing.T) {
	result := encodeNativeTools(nil, nil, ProviderOptions{EnableGoogleSearch: true})

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result.Tools))
	}

	m := jsonRoundTrip(t, result)
	toolsArr := m["tools"].([]any)
	searchTool := toolsArr[0].(map[string]any)
	searchCfg, ok := searchTool["googleSearch"].(map[string]any)
	if !ok {
		t.Fatalf("expected googleSearch key, got %v", searchTool)
	}
	if len(searchCfg) != 0 {
		t.Errorf("expected empty googleSearch config, got %v", searchCfg)
	}
}

func TestEncodeNativeTools_GoogleSearchWithFullConfig(t *testing.T) {
	threshold := 0.65
	opts := ProviderOptions{
		EnableGoogleSearch: true,
		GoogleSearchConfig: &GoogleSearchConfig{
			DynamicRetrievalThreshold: &threshold,
			SearchTypes:               []string{"web", "image"},
			TimeRangeFilter: &TimeRangeFilter{
				StartTime: "2024-01-01T00:00:00Z",
				EndTime:   "2024-12-31T23:59:59Z",
			},
		},
	}

	result := encodeNativeTools(nil, nil, opts)

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result.Tools))
	}

	m := jsonRoundTrip(t, result)
	toolsArr := m["tools"].([]any)
	searchTool := toolsArr[0].(map[string]any)
	searchCfg := searchTool["googleSearch"].(map[string]any)

	// Dynamic retrieval config.
	dynCfg := searchCfg["dynamic_retrieval_config"].(map[string]any)
	if dynCfg["mode"] != "MODE_DYNAMIC" {
		t.Errorf("expected mode=MODE_DYNAMIC, got %v", dynCfg["mode"])
	}
	if dynCfg["dynamic_threshold"] != threshold {
		t.Errorf("expected dynamic_threshold=%v, got %v", threshold, dynCfg["dynamic_threshold"])
	}

	// Search types.
	searchTypes := searchCfg["search_types"].([]any)
	if len(searchTypes) != 2 || searchTypes[0] != "web" || searchTypes[1] != "image" {
		t.Errorf("expected search_types=[web image], got %v", searchTypes)
	}

	// Time range filter.
	trf := searchCfg["time_range_filter"].(map[string]any)
	if trf["start_time"] != "2024-01-01T00:00:00Z" {
		t.Errorf("expected start_time=2024-01-01T00:00:00Z, got %v", trf["start_time"])
	}
	if trf["end_time"] != "2024-12-31T23:59:59Z" {
		t.Errorf("expected end_time=2024-12-31T23:59:59Z, got %v", trf["end_time"])
	}
}

func TestEncodeNativeTools_FunctionToolsPlusGoogleSearch(t *testing.T) {
	tools := []ai.ToolDefinition{
		{Name: "search_docs", Description: "Search documents", InputSchema: map[string]any{"type": "object"}},
		{Name: "get_user", Description: "Get user info", InputSchema: map[string]any{"type": "object"}},
	}

	result := encodeNativeTools(tools, nil, ProviderOptions{EnableGoogleSearch: true})

	if len(result.Tools) != 2 {
		t.Fatalf("expected 2 tools entries (functions + googleSearch), got %d", len(result.Tools))
	}

	m := jsonRoundTrip(t, result)
	toolsArr := m["tools"].([]any)

	// First entry: function declarations.
	funcEntry := toolsArr[0].(map[string]any)
	decls := funcEntry["functionDeclarations"].([]any)
	if len(decls) != 2 {
		t.Errorf("expected 2 function declarations, got %d", len(decls))
	}

	// Second entry: google search.
	searchEntry := toolsArr[1].(map[string]any)
	if _, ok := searchEntry["googleSearch"]; !ok {
		t.Errorf("expected googleSearch key in second tools entry, got %v", searchEntry)
	}
}

func TestEncodeNativeTools_SchemaSanitization(t *testing.T) {
	tools := []ai.ToolDefinition{
		{
			Name:        "my_tool",
			Description: "A tool with disallowed schema keys",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":    "string",
						"default": "hello",
					},
				},
				"$ref":                 "#/definitions/Foo",
				"$defs":               map[string]any{"Foo": map[string]any{"type": "string"}},
				"additionalProperties": false,
				"examples":             []any{"example1"},
			},
		},
	}

	result := encodeNativeTools(tools, nil, ProviderOptions{})

	m := jsonRoundTrip(t, result)
	toolsArr := m["tools"].([]any)
	funcEntry := toolsArr[0].(map[string]any)
	decls := funcEntry["functionDeclarations"].([]any)
	params := decls[0].(map[string]any)["parameters"].(map[string]any)

	for _, key := range []string{"$ref", "$defs", "additionalProperties", "examples"} {
		if _, ok := params[key]; ok {
			t.Errorf("disallowed key %q should have been removed from parameters", key)
		}
	}

	// Nested default should also be removed.
	props := params["properties"].(map[string]any)
	nameProp := props["name"].(map[string]any)
	if _, ok := nameProp["default"]; ok {
		t.Error("nested disallowed key 'default' should have been removed")
	}
	if nameProp["type"] != "string" {
		t.Errorf("allowed key 'type' should be preserved, got %v", nameProp["type"])
	}
}

func TestEncodeNativeTools_NilInputSchema(t *testing.T) {
	tools := []ai.ToolDefinition{
		{Name: "no_params", Description: "Tool with no parameters"},
	}

	result := encodeNativeTools(tools, nil, ProviderOptions{})

	m := jsonRoundTrip(t, result)
	toolsArr := m["tools"].([]any)
	funcEntry := toolsArr[0].(map[string]any)
	decls := funcEntry["functionDeclarations"].([]any)
	decl := decls[0].(map[string]any)

	if _, ok := decl["parameters"]; ok {
		t.Error("expected no parameters key when InputSchema is nil")
	}
}

func TestEncodeNativeTools_EmptyToolsSlice(t *testing.T) {
	result := encodeNativeTools([]ai.ToolDefinition{}, nil, ProviderOptions{})

	if len(result.Tools) != 0 {
		t.Errorf("expected no tools for empty slice, got %d", len(result.Tools))
	}
}

func TestEncodeNativeTools_FullCombination(t *testing.T) {
	// Function tools + Google Search + tool choice required.
	tools := []ai.ToolDefinition{
		{Name: "calc", Description: "Calculator", InputSchema: map[string]any{"type": "object"}},
	}
	tc := ai.ToolChoiceRequired
	threshold := 0.5
	opts := ProviderOptions{
		EnableGoogleSearch: true,
		GoogleSearchConfig: &GoogleSearchConfig{
			DynamicRetrievalThreshold: &threshold,
		},
	}

	result := encodeNativeTools(tools, &tc, opts)

	// 2 tools entries: functions + google search.
	if len(result.Tools) != 2 {
		t.Fatalf("expected 2 tools entries, got %d", len(result.Tools))
	}

	// ToolConfig present.
	if result.ToolConfig == nil {
		t.Fatal("expected toolConfig to be set")
	}

	m := jsonRoundTrip(t, result)
	toolConfig := m["toolConfig"].(map[string]any)
	fcc := toolConfig["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("expected mode=ANY, got %v", fcc["mode"])
	}
}
