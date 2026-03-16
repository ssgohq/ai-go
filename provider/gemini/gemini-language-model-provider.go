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
	core *openaichat.LanguageModel
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
		SanitizeTools:        sanitizeToolSchemas,
		ExtraToolsForRequest: extraToolsForRequest,
	})
	return &LanguageModel{core: core}
}

// ModelID returns the Gemini model identifier.
func (m *LanguageModel) ModelID() string { return m.core.ModelID() }

// Stream sends a streaming chat request and returns a channel of normalized ai.StreamEvents.
// Warnings for unsupported option combinations (e.g. TopK or Seed with Google Search) are
// injected into the first StreamEventFinish event.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	warnings := warningsForRequest(req)

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
func extraToolsForRequest(req ai.LanguageModelRequest) []map[string]any {
	opts := parseProviderOptions(req.ProviderOptions)
	if !opts.EnableGoogleSearch {
		return nil
	}

	tool := map[string]any{"type": "google_search"}

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
			tool["google_search"] = searchCfg
		}
	}

	return []map[string]any{tool}
}

// warningsForRequest returns advisory warnings for unsupported option combinations
// when Google Search grounding is enabled.
func warningsForRequest(req ai.LanguageModelRequest) []ai.Warning {
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
