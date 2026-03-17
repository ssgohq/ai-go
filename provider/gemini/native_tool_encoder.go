package gemini

import "github.com/open-ai-sdk/ai-go/ai"

// nativeToolsResult holds the encoded tools and toolConfig for a native Gemini request.
type nativeToolsResult struct {
	Tools      []any `json:"tools,omitempty"`
	ToolConfig any   `json:"toolConfig,omitempty"`
}

// encodeNativeTools converts tool definitions, tool choice, and provider options
// into the native Gemini tools and toolConfig format.
//
// Function declarations are grouped into a single tools entry with a
// "functionDeclarations" array. Google Search grounding (when enabled) is
// added as a separate entry in the tools array.
func encodeNativeTools(tools []ai.ToolDefinition, toolChoice *ai.ToolChoice, opts ProviderOptions) nativeToolsResult {
	var result nativeToolsResult

	// Build function declarations.
	if len(tools) > 0 {
		decls := make([]map[string]any, len(tools))
		for i, td := range tools {
			decl := map[string]any{
				"name":        td.Name,
				"description": td.Description,
			}
			if td.InputSchema != nil {
				decl["parameters"] = sanitizeMap(td.InputSchema)
			}
			decls[i] = decl
		}
		result.Tools = append(result.Tools, map[string]any{
			"functionDeclarations": decls,
		})
	}

	// Google Search grounding tool.
	if opts.EnableGoogleSearch {
		searchTool := buildGoogleSearchTool(opts.GoogleSearchConfig)
		result.Tools = append(result.Tools, searchTool)
	}

	// Tool config (function calling mode).
	if toolChoice != nil {
		result.ToolConfig = encodeToolConfig(toolChoice)
	}

	return result
}

// buildGoogleSearchTool builds the native Gemini googleSearch tool entry.
func buildGoogleSearchTool(cfg *GoogleSearchConfig) map[string]any {
	searchCfg := map[string]any{}

	if cfg != nil {
		if cfg.DynamicRetrievalThreshold != nil {
			searchCfg["dynamic_retrieval_config"] = map[string]any{
				"mode":              "MODE_DYNAMIC",
				"dynamic_threshold": *cfg.DynamicRetrievalThreshold,
			}
		}
		if len(cfg.SearchTypes) > 0 {
			searchCfg["search_types"] = cfg.SearchTypes
		}
		if cfg.TimeRangeFilter != nil {
			searchCfg["time_range_filter"] = map[string]any{
				"start_time": cfg.TimeRangeFilter.StartTime,
				"end_time":   cfg.TimeRangeFilter.EndTime,
			}
		}
	}

	return map[string]any{"googleSearch": searchCfg}
}

// encodeToolConfig maps an ai.ToolChoice to the native Gemini toolConfig format.
func encodeToolConfig(tc *ai.ToolChoice) map[string]any {
	fcc := map[string]any{}

	switch tc.Type {
	case "auto":
		fcc["mode"] = "AUTO"
	case "none":
		fcc["mode"] = "NONE"
	case "required":
		fcc["mode"] = "ANY"
	case "tool":
		fcc["mode"] = "ANY"
		if tc.ToolName != "" {
			fcc["allowedFunctionNames"] = []string{tc.ToolName}
		}
	}

	return map[string]any{
		"functionCallingConfig": fcc,
	}
}
