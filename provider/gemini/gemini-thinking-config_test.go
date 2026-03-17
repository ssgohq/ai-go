package gemini

import (
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

func TestThinkingConfig_NilWhenNotSet(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hello")},
	}

	fields := extraBodyFieldsForRequest(req)
	if fields != nil {
		t.Errorf("expected nil body fields when ThinkingConfig not set, got %v", fields)
	}
}

func TestThinkingConfig_NilProviderOptions(t *testing.T) {
	req := ai.LanguageModelRequest{
		Messages:        []ai.Message{ai.UserMessage("hello")},
		ProviderOptions: map[string]any{"gemini": ProviderOptions{}},
	}

	fields := extraBodyFieldsForRequest(req)
	if fields != nil {
		t.Errorf("expected nil body fields when ThinkingConfig is nil, got %v", fields)
	}
}

// extractThinkingConfig is a test helper that extracts the thinking_config
// map from the nested google.thinking_config structure.
func extractThinkingConfig(t *testing.T, fields map[string]any) map[string]any {
	t.Helper()
	google, ok := fields["google"].(map[string]any)
	if !ok {
		t.Fatalf("expected google map, got %T", fields["google"])
	}
	tc, ok := google["thinking_config"].(map[string]any)
	if !ok {
		t.Fatalf("expected thinking_config map, got %T", google["thinking_config"])
	}
	return tc
}

func TestThinkingConfig_WithThinkingBudget(t *testing.T) {
	budget := 1024
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("think hard")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{
				ThinkingConfig: &ThinkingConfig{
					ThinkingBudget: &budget,
				},
			},
		},
	}

	fields := extraBodyFieldsForRequest(req)
	if fields == nil {
		t.Fatal("expected body fields with ThinkingConfig set, got nil")
	}

	tc := extractThinkingConfig(t, fields)
	if tc["thinking_budget"] != budget {
		t.Errorf("expected thinking_budget=%d, got %v", budget, tc["thinking_budget"])
	}
	if _, ok := tc["include_thoughts"]; ok {
		t.Error("expected no include_thoughts key when not set")
	}
	if _, ok := tc["thinking_level"]; ok {
		t.Error("expected no thinking_level key when not set")
	}
}

func TestThinkingConfig_WithIncludeThoughts(t *testing.T) {
	includeThoughts := true
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("think and share")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{
				ThinkingConfig: &ThinkingConfig{
					IncludeThoughts: &includeThoughts,
				},
			},
		},
	}

	fields := extraBodyFieldsForRequest(req)
	if fields == nil {
		t.Fatal("expected body fields, got nil")
	}

	tc := extractThinkingConfig(t, fields)
	if tc["include_thoughts"] != true {
		t.Errorf("expected include_thoughts=true, got %v", tc["include_thoughts"])
	}
}

func TestThinkingConfig_WithThinkingLevel(t *testing.T) {
	levels := []string{"minimal", "low", "medium", "high"}

	for _, level := range levels {
		t.Run(level, func(t *testing.T) {
			req := ai.LanguageModelRequest{
				Messages: []ai.Message{ai.UserMessage("hello")},
				ProviderOptions: map[string]any{
					"gemini": ProviderOptions{
						ThinkingConfig: &ThinkingConfig{
							ThinkingLevel: level,
						},
					},
				},
			}

			fields := extraBodyFieldsForRequest(req)
			if fields == nil {
				t.Fatal("expected body fields, got nil")
			}

			tc := extractThinkingConfig(t, fields)
			if tc["thinking_level"] != level {
				t.Errorf("expected thinking_level=%s, got %v", level, tc["thinking_level"])
			}
		})
	}
}

func TestThinkingConfig_AllFieldsSet(t *testing.T) {
	budget := 2048
	includeThoughts := false
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("reason carefully")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{
				ThinkingConfig: &ThinkingConfig{
					ThinkingBudget:  &budget,
					IncludeThoughts: &includeThoughts,
					ThinkingLevel:   "high",
				},
			},
		},
	}

	fields := extraBodyFieldsForRequest(req)
	if fields == nil {
		t.Fatal("expected body fields, got nil")
	}

	tc := extractThinkingConfig(t, fields)
	if tc["thinking_budget"] != budget {
		t.Errorf("expected thinking_budget=%d, got %v", budget, tc["thinking_budget"])
	}
	if tc["include_thoughts"] != false {
		t.Errorf("expected include_thoughts=false, got %v", tc["include_thoughts"])
	}
	if tc["thinking_level"] != "high" {
		t.Errorf("expected thinking_level=high, got %v", tc["thinking_level"])
	}
}

func TestThinkingConfig_EmptyStruct_ReturnsNil(t *testing.T) {
	// ThinkingConfig with no fields set should return nil (nothing to inject).
	req := ai.LanguageModelRequest{
		Messages: []ai.Message{ai.UserMessage("hello")},
		ProviderOptions: map[string]any{
			"gemini": ProviderOptions{
				ThinkingConfig: &ThinkingConfig{},
			},
		},
	}

	fields := extraBodyFieldsForRequest(req)
	if fields != nil {
		t.Errorf("expected nil for empty ThinkingConfig, got %v", fields)
	}
}
