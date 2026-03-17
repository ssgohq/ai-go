package ai_test

import (
	"encoding/json"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

func TestPruneMessages(t *testing.T) {
	reasoning := ai.ContentPart{Type: ai.ContentPartTypeReasoning, ReasoningText: "thinking..."}
	text := ai.TextPart("hello")
	toolCall := ai.ToolCallPart("tc1", "search", json.RawMessage(`{"q":"go"}`))
	toolResult := ai.ToolResultPart("tc1", "search", `{"ok":true}`)

	tests := []struct {
		name     string
		messages []ai.Message
		opts     ai.PruneOptions
		wantLen  int
		check    func(t *testing.T, result []ai.Message)
	}{
		{
			name:     "empty input returns nil",
			messages: nil,
			opts:     ai.PruneOptions{},
			wantLen:  0,
			check: func(t *testing.T, result []ai.Message) {
				if result != nil {
					t.Error("expected nil for empty input")
				}
			},
		},
		{
			name: "default options returns copy unchanged",
			messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{text}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text}},
			},
			opts:    ai.PruneOptions{},
			wantLen: 2,
			check: func(t *testing.T, result []ai.Message) {
				if len(result[1].Content) != 2 {
					t.Errorf("expected 2 parts in assistant msg, got %d", len(result[1].Content))
				}
			},
		},
		{
			name: "reasoning all removes all reasoning parts",
			messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text}},
			},
			opts:    ai.PruneOptions{Reasoning: ai.PruneModeAll},
			wantLen: 2,
			check: func(t *testing.T, result []ai.Message) {
				for i, msg := range result {
					for _, part := range msg.Content {
						if part.Type == ai.ContentPartTypeReasoning {
							t.Errorf("msg[%d]: reasoning part should have been removed", i)
						}
					}
					if len(msg.Content) != 1 {
						t.Errorf("msg[%d]: expected 1 part, got %d", i, len(msg.Content))
					}
				}
			},
		},
		{
			name: "reasoning before-last-message preserves last only",
			messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text}},
			},
			opts:    ai.PruneOptions{Reasoning: ai.PruneModeBeforeLastMsg},
			wantLen: 3,
			check: func(t *testing.T, result []ai.Message) {
				// First two messages should have reasoning removed.
				for i := 0; i < 2; i++ {
					if len(result[i].Content) != 1 {
						t.Errorf("msg[%d]: expected 1 part, got %d", i, len(result[i].Content))
					}
				}
				// Last message should still have reasoning.
				if len(result[2].Content) != 2 {
					t.Errorf("last msg: expected 2 parts, got %d", len(result[2].Content))
				}
			},
		},
		{
			name: "toolCalls all removes tool_call tool_result and tool messages",
			messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{text, toolCall}},
				{Role: ai.RoleTool, Content: []ai.ContentPart{toolResult}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{text}},
			},
			opts:    ai.PruneOptions{ToolCalls: ai.PruneModeAll, EmptyMessages: ai.PruneModeRemove},
			wantLen: 2,
			check: func(t *testing.T, result []ai.Message) {
				// First assistant msg should have only text.
				if len(result[0].Content) != 1 {
					t.Errorf("msg[0]: expected 1 part, got %d", len(result[0].Content))
				}
				// Tool message should be removed (empty after pruning).
				// Last assistant msg preserved.
				if result[1].Role != ai.RoleAssistant {
					t.Errorf("msg[1]: expected assistant, got %s", result[1].Role)
				}
			},
		},
		{
			name: "toolCalls before-last-n-messages with N=2",
			messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{text, toolCall}},
				{Role: ai.RoleTool, Content: []ai.ContentPart{toolResult}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{text, toolCall}},
				{Role: ai.RoleTool, Content: []ai.ContentPart{toolResult}},
			},
			opts:    ai.PruneOptions{ToolCalls: ai.PruneModeBeforeLastNMsgs, N: 2},
			wantLen: 4,
			check: func(t *testing.T, result []ai.Message) {
				// First two messages (index 0,1) should have tool parts stripped.
				if len(result[0].Content) != 1 {
					t.Errorf("msg[0]: expected 1 part (text only), got %d", len(result[0].Content))
				}
				if len(result[1].Content) != 0 {
					t.Errorf("msg[1] (tool): expected 0 parts, got %d", len(result[1].Content))
				}
				// Last two (index 2,3) should be preserved.
				if len(result[2].Content) != 2 {
					t.Errorf("msg[2]: expected 2 parts, got %d", len(result[2].Content))
				}
				if len(result[3].Content) != 1 {
					t.Errorf("msg[3]: expected 1 part, got %d", len(result[3].Content))
				}
			},
		},
		{
			name: "emptyMessages remove drops empty messages",
			messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning}},
				{Role: ai.RoleUser, Content: []ai.ContentPart{text}},
			},
			opts:    ai.PruneOptions{Reasoning: ai.PruneModeAll, EmptyMessages: ai.PruneModeRemove},
			wantLen: 1,
			check: func(t *testing.T, result []ai.Message) {
				if result[0].Role != ai.RoleUser {
					t.Errorf("expected user msg, got %s", result[0].Role)
				}
			},
		},
		{
			name: "emptyMessages keep retains empty messages",
			messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning}},
				{Role: ai.RoleUser, Content: []ai.ContentPart{text}},
			},
			opts:    ai.PruneOptions{Reasoning: ai.PruneModeAll, EmptyMessages: ai.PruneModeKeep},
			wantLen: 2,
			check: func(t *testing.T, result []ai.Message) {
				if len(result[0].Content) != 0 {
					t.Errorf("expected empty content, got %d parts", len(result[0].Content))
				}
			},
		},
		{
			name: "original slice not mutated",
			messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text}},
			},
			opts:    ai.PruneOptions{Reasoning: ai.PruneModeAll},
			wantLen: 1,
			check: func(t *testing.T, _ []ai.Message) {
				// Checked after the test body below.
			},
		},
		{
			name: "combined reasoning and toolCalls pruning",
			messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text, toolCall}},
				{Role: ai.RoleTool, Content: []ai.ContentPart{toolResult}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{reasoning, text}},
			},
			opts: ai.PruneOptions{
				Reasoning:     ai.PruneModeAll,
				ToolCalls:     ai.PruneModeAll,
				EmptyMessages: ai.PruneModeRemove,
			},
			wantLen: 2,
			check: func(t *testing.T, result []ai.Message) {
				// Both assistants should only have text.
				for i, msg := range result {
					if len(msg.Content) != 1 || msg.Content[0].Type != ai.ContentPartTypeText {
						t.Errorf("msg[%d]: expected single text part, got %d parts", i, len(msg.Content))
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Snapshot original for mutation check.
			var origParts int
			if len(tt.messages) > 0 {
				origParts = len(tt.messages[0].Content)
			}

			result := ai.PruneMessages(tt.messages, tt.opts)

			if tt.messages == nil {
				if result != nil {
					t.Fatalf("expected nil result, got %v", result)
				}
				tt.check(t, result)
				return
			}

			if len(result) != tt.wantLen {
				t.Fatalf("expected %d messages, got %d", tt.wantLen, len(result))
			}

			tt.check(t, result)

			// Verify original not mutated.
			if len(tt.messages) > 0 && len(tt.messages[0].Content) != origParts {
				t.Errorf("original messages mutated: had %d parts, now %d", origParts, len(tt.messages[0].Content))
			}
		})
	}
}
