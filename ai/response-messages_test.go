package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

type responseMessageModel struct {
	step int
}

func (m *responseMessageModel) ModelID() string { return "response-messages" }

func (m *responseMessageModel) Stream(
	_ context.Context,
	_ ai.LanguageModelRequest,
) (<-chan ai.StreamEvent, error) {
	m.step++
	ch := make(chan ai.StreamEvent, 8)
	if m.step == 1 {
		ch <- ai.StreamEvent{
			Type:              ai.StreamEventToolCallDelta,
			ToolCallIndex:     0,
			ToolCallID:        "tc-001",
			ToolCallName:      "add",
			ToolCallArgsDelta: `{"a":1,"b":2}`,
			ThoughtSignature:  "sig-1",
		}
		ch <- ai.StreamEvent{
			Type:  ai.StreamEventUsage,
			Usage: &ai.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3},
		}
		ch <- ai.StreamEvent{
			Type:         ai.StreamEventFinish,
			FinishReason: ai.FinishReasonToolCalls,
		}
	} else {
		ch <- ai.StreamEvent{Type: ai.StreamEventReasoningDelta, TextDelta: "I used a calculator."}
		ch <- ai.StreamEvent{Type: ai.StreamEventTextDelta, TextDelta: "The answer is 3."}
		ch <- ai.StreamEvent{
			Type:  ai.StreamEventUsage,
			Usage: &ai.Usage{PromptTokens: 4, CompletionTokens: 5, TotalTokens: 9},
		}
		ch <- ai.StreamEvent{
			Type:         ai.StreamEventFinish,
			FinishReason: ai.FinishReasonStop,
		}
	}
	close(ch)
	return ch, nil
}

type repairToolCallModel struct {
	step int
}

func (m *repairToolCallModel) ModelID() string { return "repair-tool-call" }

func (m *repairToolCallModel) Stream(
	_ context.Context,
	_ ai.LanguageModelRequest,
) (<-chan ai.StreamEvent, error) {
	m.step++
	ch := make(chan ai.StreamEvent, 8)
	if m.step == 1 {
		ch <- ai.StreamEvent{
			Type:              ai.StreamEventToolCallDelta,
			ToolCallIndex:     0,
			ToolCallID:        "tc-repair",
			ToolCallName:      "ADD",
			ToolCallArgsDelta: `{"a":1,"b":2}`,
		}
		ch <- ai.StreamEvent{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonToolCalls}
	} else {
		ch <- ai.StreamEvent{Type: ai.StreamEventTextDelta, TextDelta: "Done"}
		ch <- ai.StreamEvent{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop}
	}
	close(ch)
	return ch, nil
}

func TestResponseMessagesForStep_UsesToModelOutput(t *testing.T) {
	step := ai.StepOutput{
		Reasoning: "Need the calculator first.",
		Text:      "Calling add.",
		ToolCalls: []ai.ToolCallOutput{{
			ID:               "tc-1",
			Name:             "add",
			Args:             json.RawMessage(`{"a":1,"b":2}`),
			ThoughtSignature: "sig-1",
		}},
		ToolResults: []ai.ToolResult{{
			ID:     "tc-1",
			Name:   "add",
			Output: `{"result":3}`,
		}},
	}
	tools := &ai.ToolSet{
		Definitions: []ai.ToolDefinition{{
			Name: "add",
			ToModelOutput: func(result string) string {
				return `{"summary":"3"}`
			},
		}},
	}

	messages := ai.ResponseMessagesForStep(step, tools)
	if len(messages) != 2 {
		t.Fatalf("expected 2 response messages, got %d", len(messages))
	}
	if messages[0].Role != ai.RoleAssistant {
		t.Fatalf("expected assistant response message, got %s", messages[0].Role)
	}
	if len(messages[0].Content) != 3 {
		t.Fatalf("expected reasoning, text, and tool call parts, got %d", len(messages[0].Content))
	}
	if messages[0].Content[0].Type != ai.ContentPartTypeReasoning ||
		messages[0].Content[0].ReasoningText != "Need the calculator first." {
		t.Fatalf("unexpected reasoning part: %+v", messages[0].Content[0])
	}
	if messages[0].Content[2].ThoughtSignature != "sig-1" {
		t.Fatalf("expected thought signature to be preserved, got %+v", messages[0].Content[2])
	}
	if messages[1].Content[0].ToolResultOutput != `{"summary":"3"}` {
		t.Fatalf("expected ToModelOutput to be applied, got %q", messages[1].Content[0].ToolResultOutput)
	}
}

func TestGenerateText_ResponseMessagesAndCallbacks(t *testing.T) {
	model := &responseMessageModel{}
	var stepEvents []ai.StepFinishEvent
	var finishEvent ai.FinishEvent

	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("What is 1+2?")},
		Tools: &ai.ToolSet{
			Definitions: []ai.ToolDefinition{{
				Name: "add",
				ToModelOutput: func(result string) string {
					return `{"rendered":"3"}`
				},
			}},
			Executor: &addExecutor{},
		},
		MaxSteps: 5,
		OnStepFinish: func(event ai.StepFinishEvent) {
			stepEvents = append(stepEvents, event)
		},
		OnFinish: func(event ai.FinishEvent) {
			finishEvent = event
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(result.Steps))
	}
	if string(result.Steps[0].ToolCalls[0].Args) != `{"a":1,"b":2}` {
		t.Fatalf("expected tool call args to be preserved, got %q", result.Steps[0].ToolCalls[0].Args)
	}
	if result.Steps[0].ToolCalls[0].ThoughtSignature != "sig-1" {
		t.Fatalf("expected thought signature to be preserved, got %q", result.Steps[0].ToolCalls[0].ThoughtSignature)
	}
	if len(result.Steps[0].Response.Messages) != 2 {
		t.Fatalf("expected step 0 response messages, got %d", len(result.Steps[0].Response.Messages))
	}
	if len(result.Steps[1].Response.Messages) != 1 {
		t.Fatalf("expected step 1 response messages, got %d", len(result.Steps[1].Response.Messages))
	}
	if len(result.Response.Messages) != 3 {
		t.Fatalf("expected aggregated response messages, got %d", len(result.Response.Messages))
	}
	if result.Response.Messages[1].Content[0].ToolResultOutput != `{"rendered":"3"}` {
		t.Fatalf(
			"expected transformed tool output in response messages, got %q",
			result.Response.Messages[1].Content[0].ToolResultOutput,
		)
	}

	if len(stepEvents) != 2 {
		t.Fatalf("expected 2 step finish callbacks, got %d", len(stepEvents))
	}
	if len(stepEvents[0].Response.Messages) != 2 {
		t.Fatalf("expected step finish response messages, got %d", len(stepEvents[0].Response.Messages))
	}
	if stepEvents[1].Reasoning != "I used a calculator." {
		t.Fatalf("expected reasoning on step finish, got %q", stepEvents[1].Reasoning)
	}
	if finishEvent.TotalUsage.TotalTokens != 12 {
		t.Fatalf("expected total usage across steps, got %d", finishEvent.TotalUsage.TotalTokens)
	}
	if len(finishEvent.Response.Messages) != 3 {
		t.Fatalf("expected finish response messages, got %d", len(finishEvent.Response.Messages))
	}
}

func TestGenerateText_ExperimentalRepairToolCall(t *testing.T) {
	model := &repairToolCallModel{}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("What is 1+2?")},
		Tools: &ai.ToolSet{
			Definitions: []ai.ToolDefinition{{Name: "add"}},
			Executor:    &addExecutor{},
		},
		MaxSteps: 5,
		ExperimentalRepairToolCall: func(_ context.Context, input ai.RepairToolCallInput) (*ai.ToolCallOutput, error) {
			var noSuchToolErr *ai.NoSuchToolError
			if !errors.As(input.Error, &noSuchToolErr) {
				t.Fatalf("expected NoSuchToolError, got %T", input.Error)
			}
			if input.ToolCall.Name != "ADD" {
				t.Fatalf("expected original tool name, got %q", input.ToolCall.Name)
			}
			return &ai.ToolCallOutput{
				Name: "add",
				Args: input.ToolCall.Args,
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Steps[0].ToolCalls[0].Name != "add" {
		t.Fatalf("expected repaired tool name in step output, got %q", result.Steps[0].ToolCalls[0].Name)
	}
	if result.Steps[0].Response.Messages[0].Content[0].ToolCallName != "add" {
		t.Fatalf(
			"expected repaired tool name in response message, got %q",
			result.Steps[0].Response.Messages[0].Content[0].ToolCallName,
		)
	}
}
