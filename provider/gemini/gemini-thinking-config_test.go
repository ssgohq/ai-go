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

	tc, ok := fields["thinkingConfig"].(map[string]any)
	if !ok {
		t.Fatalf("expected thinkingConfig map, got %T", fields["thinkingConfig"])
	}
	if tc["thinkingBudget"] != budget {
		t.Errorf("expected thinkingBudget=%d, got %v", budget, tc["thinkingBudget"])
	}
	if _, ok := tc["includeThoughts"]; ok {
		t.Error("expected no includeThoughts key when not set")
	}
	if _, ok := tc["thinkingLevel"]; ok {
		t.Error("expected no thinkingLevel key when not set")
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

	tc, ok := fields["thinkingConfig"].(map[string]any)
	if !ok {
		t.Fatalf("expected thinkingConfig map, got %T", fields["thinkingConfig"])
	}
	if tc["includeThoughts"] != true {
		t.Errorf("expected includeThoughts=true, got %v", tc["includeThoughts"])
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

			tc, ok := fields["thinkingConfig"].(map[string]any)
			if !ok {
				t.Fatalf("expected thinkingConfig map, got %T", fields["thinkingConfig"])
			}
			if tc["thinkingLevel"] != level {
				t.Errorf("expected thinkingLevel=%s, got %v", level, tc["thinkingLevel"])
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

	tc, ok := fields["thinkingConfig"].(map[string]any)
	if !ok {
		t.Fatalf("expected thinkingConfig map, got %T", fields["thinkingConfig"])
	}
	if tc["thinkingBudget"] != budget {
		t.Errorf("expected thinkingBudget=%d, got %v", budget, tc["thinkingBudget"])
	}
	if tc["includeThoughts"] != false {
		t.Errorf("expected includeThoughts=false, got %v", tc["includeThoughts"])
	}
	if tc["thinkingLevel"] != "high" {
		t.Errorf("expected thinkingLevel=high, got %v", tc["thinkingLevel"])
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
