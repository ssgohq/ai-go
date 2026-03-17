package gemini

import (
	"context"
	"time"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/provider/internal/openaichat"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta/openai"

// LanguageModel implements ai.LanguageModel for the Gemini OpenAI-compatible API.
type LanguageModel struct {
	modelID string
	core    *openaichat.LanguageModel
}

// Config holds options for constructing a Gemini LanguageModel or EmbeddingModel.
type Config struct {
	APIKey               string
	BaseURL              string        // optional; defaults to Gemini production endpoint
	Timeout              time.Duration // optional; defaults to 120s for LLM, 60s for embedding
	OutputDimensionality int           // optional; embedding output dimensions (768, 1536, 3072)
}

// NewLanguageModel creates a Gemini-backed ai.LanguageModel.
func NewLanguageModel(modelID string, cfg Config) *LanguageModel {
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	core := openaichat.NewLanguageModel(openaichat.ModelConfig{
		ModelID:      modelID,
		ProviderName: "gemini",
		BaseURL:      base,
		APIKey:       cfg.APIKey,
		Timeout:      cfg.Timeout,
		Capabilities: openaichat.CapabilityFlags{
			SupportsStructuredOutput: true,
			SupportsStreamUsage:      true,
		},
		SanitizeTools: sanitizeToolSchemas,
		ExtraToolsForRequest: func(req ai.LanguageModelRequest) []map[string]any {
			return extraToolsForRequest(modelID, req)
		},
		ExtraBodyFieldsForRequest: extraBodyFieldsForRequest,
		TransformRequestBody:      mergeGoogleSearchTools,
	})
	return &LanguageModel{modelID: modelID, core: core}
}

// ModelID returns the Gemini model identifier.
func (m *LanguageModel) ModelID() string { return m.core.ModelID() }

// Stream sends a streaming chat request and returns a channel of normalized ai.StreamEvents.
// Warnings for unsupported option combinations (e.g. TopK or Seed with Google Search)
// are injected into the first StreamEventFinish event.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	warnings := warningsForRequest(m.modelID, req)

	coreCh, err := m.core.Stream(ctx, req)
	if err != nil {
		return nil, err
	}

	if len(warnings) == 0 {
		return coreCh, nil
	}

	// Wrap the core channel to inject warnings into the first finish event.
	out := make(chan ai.StreamEvent, 64)
	go func() {
		defer close(out)
		finishInjected := false
		for ev := range coreCh {
			if !finishInjected && ev.Type == ai.StreamEventFinish {
				ev.Warnings = append(warnings, ev.Warnings...)
				finishInjected = true
			}
			out <- ev
		}
	}()
	return out, nil
}

// extraToolsForRequest returns Gemini-specific built-in tools based on provider options.
// Google Search grounding is passed via extra body fields (not the tools array) because
// the OpenAI-compatible endpoint only supports {"type":"function"} tools. The native
// google_search tool is injected via extraBodyFieldsForRequest instead.
func extraToolsForRequest(modelID string, req ai.LanguageModelRequest) []map[string]any {
	// google_search is no longer sent as a tool — it goes via extra body fields.
	return nil
}

// extraBodyFieldsForRequest returns provider-specific top-level request body fields
// based on provider options. Handles thinkingConfig.
// Google Search is handled separately via TransformRequestBody to merge into tools.
// Wired via openaichat.ModelConfig.ExtraBodyFieldsForRequest in NewLanguageModel.
func extraBodyFieldsForRequest(req ai.LanguageModelRequest) map[string]any {
	opts := parseProviderOptions(req.ProviderOptions)

	result := make(map[string]any)

	// Thinking config — the OpenAI-compatible endpoint expects Gemini-specific
	// fields under a top-level "google" key (matching the extra_body convention).
	if opts.ThinkingConfig != nil {
		cfg := opts.ThinkingConfig
		thinkingMap := map[string]any{}
		if cfg.ThinkingBudget != nil {
			thinkingMap["thinking_budget"] = *cfg.ThinkingBudget
		}
		if cfg.IncludeThoughts != nil {
			thinkingMap["include_thoughts"] = *cfg.IncludeThoughts
		}
		if cfg.ThinkingLevel != "" {
			thinkingMap["thinking_level"] = cfg.ThinkingLevel
		}
		if len(thinkingMap) > 0 {
			google, _ := result["google"].(map[string]any)
			if google == nil {
				google = map[string]any{}
			}
			google["thinking_config"] = thinkingMap
			result["google"] = google
		}
	}

	// Google Search grounding — append to the tools array (not replace).
	// The OpenAI-compatible endpoint accepts the camelCase Gemini tool format
	// {"googleSearch":{}} alongside function tools in the tools array.
	if opts.EnableGoogleSearch {
		searchTool := map[string]any{"googleSearch": map[string]any{}}

		if cfg := opts.GoogleSearchConfig; cfg != nil {
			searchCfg := map[string]any{}
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
			if len(searchCfg) > 0 {
				searchTool["googleSearch"] = searchCfg
			}
		}

		// Store search tool in a temporary key; mergeGoogleSearchTools will
		// move it into the actual tools array.
		result["_google_search_tool"] = searchTool
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// mergeGoogleSearchTools is a TransformRequestBody hook that moves
// the _google_search_tool entry into the tools array and removes the temp key.
func mergeGoogleSearchTools(body map[string]any) map[string]any {
	searchTool, ok := body["_google_search_tool"]
	if !ok {
		return body
	}
	delete(body, "_google_search_tool")

	// Append to existing tools array or create a new one.
	if existing, ok := body["tools"].([]any); ok {
		body["tools"] = append(existing, searchTool)
	} else if existing, ok := body["tools"].([]map[string]any); ok {
		asAny := make([]any, len(existing)+1)
		for i, t := range existing {
			asAny[i] = t
		}
		asAny[len(existing)] = searchTool
		body["tools"] = asAny
	} else {
		body["tools"] = []any{searchTool}
	}
	return body
}

// warningsForRequest returns advisory warnings for unsupported option combinations.
// Warns when Google Search is enabled with unsupported settings (topK, seed).
func warningsForRequest(modelID string, req ai.LanguageModelRequest) []ai.Warning {
	opts := parseProviderOptions(req.ProviderOptions)
	if !opts.EnableGoogleSearch {
		return nil
	}

	var warnings []ai.Warning

	if req.Settings.TopK != nil {
		warnings = append(warnings, ai.Warning{
			Type:    "unsupported-setting",
			Setting: "topK",
			Message: "topK is not supported when Google Search grounding is enabled and will be ignored",
		})
	}
	if req.Settings.Seed != nil {
		warnings = append(warnings, ai.Warning{
			Type:    "unsupported-setting",
			Setting: "seed",
			Message: "seed is not supported when Google Search grounding is enabled (grounding is non-deterministic)",
		})
	}
	return warnings
}
