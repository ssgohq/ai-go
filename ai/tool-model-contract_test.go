package ai_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// --- ToolDefinition ---

func TestToolDefinition_Fields(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{"type": "string"},
		},
		"required": []string{"query"},
	}
	td := ai.ToolDefinition{
		Name:        "web_search",
		Description: "Search the web for information",
		InputSchema: schema,
	}
	if td.Name != "web_search" {
		t.Errorf("unexpected Name: %s", td.Name)
	}
	if td.Description == "" {
		t.Error("expected non-empty Description")
	}
	if td.InputSchema == nil {
		t.Error("expected non-nil InputSchema")
	}
	if td.InputSchema["type"] != "object" {
		t.Errorf("unexpected schema type: %v", td.InputSchema["type"])
	}
}

// --- ToolChoice ---

func TestToolChoiceAuto(t *testing.T) {
	tc := ai.ToolChoiceAuto
	if tc.Type != "auto" {
		t.Errorf("expected auto, got %s", tc.Type)
	}
	if tc.ToolName != "" {
		t.Errorf("expected empty ToolName, got %s", tc.ToolName)
	}
}

func TestToolChoiceNone(t *testing.T) {
	tc := ai.ToolChoiceNone
	if tc.Type != "none" {
		t.Errorf("expected none, got %s", tc.Type)
	}
}

func TestToolChoiceRequired(t *testing.T) {
	tc := ai.ToolChoiceRequired
	if tc.Type != "required" {
		t.Errorf("expected required, got %s", tc.Type)
	}
}

func TestToolChoiceSpecific(t *testing.T) {
	tc := ai.ToolChoiceSpecific("web_search")
	if tc.Type != "tool" {
		t.Errorf("expected tool, got %s", tc.Type)
	}
	if tc.ToolName != "web_search" {
		t.Errorf("expected web_search, got %s", tc.ToolName)
	}
}

// --- ToolChoice propagation through GenerateText ---

// toolChoiceCaptureModel captures the ToolChoice from the LanguageModelRequest.
type toolChoiceCaptureModel struct {
	capturedChoice *ai.ToolChoice
}

func (m *toolChoiceCaptureModel) ModelID() string { return "tc-capture" }

func (m *toolChoiceCaptureModel) Stream(
	_ context.Context,
	req ai.LanguageModelRequest,
) (<-chan ai.StreamEvent, error) {
	m.capturedChoice = req.ToolChoice
	ch := make(chan ai.StreamEvent, 2)
	ch <- ai.StreamEvent{Type: ai.StreamEventTextDelta, TextDelta: "done"}
	ch <- ai.StreamEvent{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop}
	close(ch)
	return ch, nil
}

func TestToolChoicePropagation_Auto(t *testing.T) {
	model := &toolChoiceCaptureModel{}
	tc := ai.ToolChoiceAuto
	_, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:      model,
		Messages:   []ai.Message{ai.UserMessage("hi")},
		ToolChoice: &tc,
		Tools: &ai.ToolSet{
			Definitions: []ai.ToolDefinition{{Name: "t", InputSchema: map[string]any{"type": "object"}}},
			Executor:    nil,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.capturedChoice == nil {
		t.Fatal("ToolChoice not propagated")
	}
	if model.capturedChoice.Type != "auto" {
		t.Errorf("expected auto, got %s", model.capturedChoice.Type)
	}
}

func TestToolChoicePropagation_Required(t *testing.T) {
	model := &toolChoiceCaptureModel{}
	tc := ai.ToolChoiceRequired
	_, _ = ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:      model,
		Messages:   []ai.Message{ai.UserMessage("hi")},
		ToolChoice: &tc,
		Tools: &ai.ToolSet{
			Definitions: []ai.ToolDefinition{{Name: "t", InputSchema: map[string]any{"type": "object"}}},
		},
	})
	if model.capturedChoice == nil || model.capturedChoice.Type != "required" {
		t.Errorf("expected required, got %v", model.capturedChoice)
	}
}

func TestToolChoicePropagation_Specific(t *testing.T) {
	model := &toolChoiceCaptureModel{}
	tc := ai.ToolChoiceSpecific("my_tool")
	_, _ = ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:      model,
		Messages:   []ai.Message{ai.UserMessage("hi")},
		ToolChoice: &tc,
		Tools: &ai.ToolSet{
			Definitions: []ai.ToolDefinition{{Name: "my_tool", InputSchema: map[string]any{"type": "object"}}},
		},
	})
	if model.capturedChoice == nil {
		t.Fatal("ToolChoice not propagated")
	}
	if model.capturedChoice.Type != "tool" {
		t.Errorf("expected tool, got %s", model.capturedChoice.Type)
	}
	if model.capturedChoice.ToolName != "my_tool" {
		t.Errorf("expected my_tool, got %s", model.capturedChoice.ToolName)
	}
}

func TestToolChoicePropagation_Nil(t *testing.T) {
	model := &toolChoiceCaptureModel{}
	_, _ = ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:      model,
		Messages:   []ai.Message{ai.UserMessage("hi")},
		ToolChoice: nil,
	})
	if model.capturedChoice != nil {
		t.Errorf("expected nil ToolChoice, got %+v", model.capturedChoice)
	}
}

// --- Multi-step tool loop ---

// multiStepModel drives a two-step tool loop:
// step 1 emits a tool call, step 2 emits text and stops.
type multiStepModel struct {
	step int
}

func (m *multiStepModel) ModelID() string { return "multi-step" }

func (m *multiStepModel) Stream(
	_ context.Context,
	req ai.LanguageModelRequest,
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
		}
		ch <- ai.StreamEvent{
			Type:         ai.StreamEventFinish,
			FinishReason: ai.FinishReasonToolCalls,
		}
	} else {
		ch <- ai.StreamEvent{Type: ai.StreamEventTextDelta, TextDelta: "The answer is 3."}
		ch <- ai.StreamEvent{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop}
	}
	close(ch)
	return ch, nil
}

type addExecutor struct{}

func (e *addExecutor) Execute(_ context.Context, name, argsJSON string) (string, error) {
	var args struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", err
	}
	result := args.A + args.B
	out, _ := json.Marshal(map[string]int{"result": result})
	return string(out), nil
}

func TestToolLoop_TwoStep(t *testing.T) {
	model := &multiStepModel{}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("What is 1+2?")},
		Tools: &ai.ToolSet{
			Definitions: []ai.ToolDefinition{{
				Name:        "add",
				Description: "Add two integers",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"a": map[string]any{"type": "integer"},
						"b": map[string]any{"type": "integer"},
					},
				},
			}},
			Executor: &addExecutor{},
		},
		MaxSteps: 5,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "The answer is 3." {
		t.Errorf("unexpected Text: %q", result.Text)
	}
	if len(result.Steps) != 2 {
		t.Errorf("expected 2 steps, got %d", len(result.Steps))
	}
	if result.FinishReason != ai.FinishReasonStop {
		t.Errorf("expected stop, got %s", result.FinishReason)
	}
}

func TestToolLoop_StopCondition_HasToolCall(t *testing.T) {
	model := &multiStepModel{}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("compute")},
		Tools: &ai.ToolSet{
			Definitions: []ai.ToolDefinition{{
				Name:        "add",
				InputSchema: map[string]any{"type": "object"},
			}},
			Executor: &addExecutor{},
		},
		StopWhen: ai.HasToolCall("add"),
		MaxSteps: 10,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// HasToolCall("add") stops after step 1 once add was called.
	// The loop exits before step 2, so Text is empty.
	if len(result.Steps) != 1 {
		t.Errorf("expected 1 step with HasToolCall stop, got %d", len(result.Steps))
	}
}

// --- CallSettings propagation ---

type callSettingsCaptureModel struct {
	capturedSettings ai.CallSettings
}

func (m *callSettingsCaptureModel) ModelID() string { return "settings-capture" }

func (m *callSettingsCaptureModel) Stream(
	_ context.Context,
	req ai.LanguageModelRequest,
) (<-chan ai.StreamEvent, error) {
	m.capturedSettings = req.Settings
	ch := make(chan ai.StreamEvent, 2)
	ch <- ai.StreamEvent{Type: ai.StreamEventTextDelta, TextDelta: "ok"}
	ch <- ai.StreamEvent{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop}
	close(ch)
	return ch, nil
}

func TestCallSettings_AllFieldsPropagated(t *testing.T) {
	model := &callSettingsCaptureModel{}
	temp := float32(0.7)
	topP := float32(0.9)
	topK := 40
	seed := 42
	_, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("hi")},
		Settings: ai.CallSettings{
			Temperature:   &temp,
			MaxTokens:     512,
			TopP:          &topP,
			TopK:          &topK,
			Seed:          &seed,
			StopSequences: []string{"STOP", "END"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	s := model.capturedSettings
	if s.Temperature == nil || *s.Temperature != 0.7 {
		t.Errorf("Temperature mismatch: %v", s.Temperature)
	}
	if s.MaxTokens != 512 {
		t.Errorf("MaxTokens mismatch: %d", s.MaxTokens)
	}
	if s.TopP == nil || *s.TopP != 0.9 {
		t.Errorf("TopP mismatch: %v", s.TopP)
	}
	if s.TopK == nil || *s.TopK != 40 {
		t.Errorf("TopK mismatch: %v", s.TopK)
	}
	if s.Seed == nil || *s.Seed != 42 {
		t.Errorf("Seed mismatch: %v", s.Seed)
	}
	if len(s.StopSequences) != 2 || s.StopSequences[0] != "STOP" {
		t.Errorf("StopSequences mismatch: %v", s.StopSequences)
	}
}
