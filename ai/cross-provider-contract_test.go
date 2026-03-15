package ai_test

import (
	"context"
	"sync"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// crossProviderMock simulates how any provider (OpenAI or Gemini) implements
// ai.LanguageModel with the same GenerateTextRequest. This test verifies that
// the same request works through the common adapter path.
type crossProviderMock struct {
	name string
	mu   sync.Mutex
	// reqs records every call in order so tests can inspect the first (non-structured-output) call.
	reqs           []ai.LanguageModelRequest
	responseEvents []ai.StreamEvent
}

func (m *crossProviderMock) ModelID() string { return m.name }

func (m *crossProviderMock) Stream(
	_ context.Context,
	req ai.LanguageModelRequest,
) (<-chan ai.StreamEvent, error) {
	m.mu.Lock()
	m.reqs = append(m.reqs, req)
	m.mu.Unlock()
	ch := make(chan ai.StreamEvent, len(m.responseEvents)+1)
	for _, ev := range m.responseEvents {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

// firstReq returns the first LanguageModelRequest the mock received.
// The engine may call the model more than once (e.g. structured output emitter),
// so the first call reflects the main generation parameters.
func (m *crossProviderMock) firstReq() ai.LanguageModelRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.reqs) == 0 {
		return ai.LanguageModelRequest{}
	}
	return m.reqs[0]
}

func newSuccessEvents(text string) []ai.StreamEvent {
	return []ai.StreamEvent{
		{Type: ai.StreamEventTextDelta, TextDelta: text},
		{
			Type:         ai.StreamEventFinish,
			FinishReason: ai.FinishReasonStop,
			Usage: &ai.Usage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		},
	}
}

// TestCrossProvider_ToolsAndChoiceForwarded verifies tools and tool choice
// are forwarded from GenerateTextRequest to the provider on the first call.
func TestCrossProvider_ToolsAndChoiceForwarded(t *testing.T) {
	toolDef := ai.ToolDefinition{
		Name:        "calculator",
		Description: "Perform arithmetic",
		InputSchema: map[string]any{"type": "object"},
	}
	tc := ai.ToolChoiceRequired

	cases := []string{"openai-mock", "gemini-mock"}
	for _, name := range cases {
		name := name
		t.Run(name, func(t *testing.T) {
			p := &crossProviderMock{
				name:           name,
				responseEvents: newSuccessEvents("result"),
			}
			_, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
				Model:    p,
				Messages: []ai.Message{ai.UserMessage("compute")},
				Tools: &ai.ToolSet{
					Definitions: []ai.ToolDefinition{toolDef},
					Executor:    nil,
				},
				ToolChoice: &tc,
				MaxSteps:   1,
			})
			if err != nil {
				t.Fatalf("[%s] unexpected error: %v", name, err)
			}

			req := p.firstReq()

			if len(req.Tools) != 1 {
				t.Fatalf("[%s] expected 1 tool, got %d", name, len(req.Tools))
			}
			if req.Tools[0].Name != "calculator" {
				t.Errorf("[%s] unexpected tool name: %s", name, req.Tools[0].Name)
			}
			if req.Tools[0].InputSchema == nil {
				t.Errorf("[%s] InputSchema not forwarded", name)
			}
			if req.ToolChoice == nil || req.ToolChoice.Type != "required" {
				t.Errorf("[%s] ToolChoice not forwarded correctly: %+v", name, req.ToolChoice)
			}
		})
	}
}

// TestCrossProvider_SettingsForwarded verifies all CallSettings fields
// are forwarded to the provider.
func TestCrossProvider_SettingsForwarded(t *testing.T) {
	cases := []string{"openai-mock", "gemini-mock"}
	for _, name := range cases {
		name := name
		t.Run(name, func(t *testing.T) {
			temp := float32(0.5)
			seed := 123
			topP := float32(0.9)
			topK := 40

			p := &crossProviderMock{
				name:           name,
				responseEvents: newSuccessEvents("done"),
			}
			_, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
				Model:    p,
				Messages: []ai.Message{ai.UserMessage("hi")},
				Settings: ai.CallSettings{
					Temperature: &temp,
					MaxTokens:   256,
					TopP:        &topP,
					TopK:        &topK,
					Seed:        &seed,
				},
				MaxSteps: 1,
			})
			if err != nil {
				t.Fatalf("[%s] unexpected error: %v", name, err)
			}

			req := p.firstReq()
			if req.Settings.Temperature == nil || *req.Settings.Temperature != 0.5 {
				t.Errorf("[%s] Temperature: got %v", name, req.Settings.Temperature)
			}
			if req.Settings.MaxTokens != 256 {
				t.Errorf("[%s] MaxTokens: got %d", name, req.Settings.MaxTokens)
			}
			if req.Settings.TopP == nil || *req.Settings.TopP != 0.9 {
				t.Errorf("[%s] TopP: got %v", name, req.Settings.TopP)
			}
			if req.Settings.TopK == nil || *req.Settings.TopK != 40 {
				t.Errorf("[%s] TopK: got %v", name, req.Settings.TopK)
			}
			if req.Settings.Seed == nil || *req.Settings.Seed != 123 {
				t.Errorf("[%s] Seed: got %v", name, req.Settings.Seed)
			}
		})
	}
}

// TestCrossProvider_FinishReasons verifies all standard finish reasons are correctly
// surfaced from any provider.
func TestCrossProvider_FinishReasons(t *testing.T) {
	cases := []struct {
		reason ai.FinishReason
		raw    string
	}{
		{ai.FinishReasonStop, "stop"},
		{ai.FinishReasonLength, "length"},
		{ai.FinishReasonToolCalls, "tool_calls"},
		{ai.FinishReasonContentFilter, "content_filter"},
		{ai.FinishReasonError, "error"},
		{ai.FinishReasonUnknown, "unknown"},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(string(tc.reason), func(t *testing.T) {
			model := &crossProviderMock{
				name: "mock",
				responseEvents: []ai.StreamEvent{
					{Type: ai.StreamEventTextDelta, TextDelta: "x"},
					{
						Type:            ai.StreamEventFinish,
						FinishReason:    tc.reason,
						RawFinishReason: tc.raw,
					},
				},
			}
			result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
				Model:    model,
				Messages: []ai.Message{ai.UserMessage("hi")},
				MaxSteps: 1,
			})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.FinishReason != tc.reason {
				t.Errorf("expected %q, got %q", tc.reason, result.FinishReason)
			}
			if result.RawFinishReason != tc.raw {
				t.Errorf("expected raw %q, got %q", tc.raw, result.RawFinishReason)
			}
		})
	}
}

// TestCrossProvider_OutputSchemaPropagated verifies the Output field is forwarded
// to the provider's first call and structured output is returned when model returns JSON.
func TestCrossProvider_OutputSchemaPropagated(t *testing.T) {
	model := &crossProviderMock{
		name: "mock",
		responseEvents: []ai.StreamEvent{
			{Type: ai.StreamEventTextDelta, TextDelta: `{"answer":"42"}`},
			{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop},
		},
	}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("return json")},
		Output:   ai.OutputObject(map[string]any{"type": "object"}),
		MaxSteps: 1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	req := model.firstReq()
	if req.Output == nil {
		t.Error("Output not propagated to provider")
	}
	if req.Output.Type != "object" {
		t.Errorf("expected output type 'object', got %q", req.Output.Type)
	}
	if result.Text == "" {
		t.Error("expected non-empty Text")
	}
}

// TestCrossProvider_ProviderOptionsPassthrough verifies provider-specific options
// reach the model on every call.
func TestCrossProvider_ProviderOptionsPassthrough(t *testing.T) {
	model := &crossProviderMock{
		name:           "openai-mock",
		responseEvents: newSuccessEvents("ok"),
	}
	_, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("hi")},
		ProviderOptions: map[string]any{
			"openai": map[string]any{"previousResponseId": "resp_123"},
		},
		MaxSteps: 1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	req := model.firstReq()
	if req.ProviderOptions == nil {
		t.Fatal("ProviderOptions not forwarded")
	}
	opts, ok := req.ProviderOptions["openai"].(map[string]any)
	if !ok {
		t.Fatalf("expected openai opts map, got %T", req.ProviderOptions["openai"])
	}
	if opts["previousResponseId"] != "resp_123" {
		t.Errorf("unexpected previousResponseId: %v", opts["previousResponseId"])
	}
}

// TestCrossProvider_UsagePropagated verifies token usage is correctly
// propagated from the provider's Usage event to GenerateTextResult.
func TestCrossProvider_UsagePropagated(t *testing.T) {
	model := &crossProviderMock{
		name: "mock",
		responseEvents: []ai.StreamEvent{
			{Type: ai.StreamEventTextDelta, TextDelta: "answer"},
			{
				Type: ai.StreamEventUsage,
				Usage: &ai.Usage{
					PromptTokens:     100,
					CompletionTokens: 50,
					TotalTokens:      150,
				},
			},
			{Type: ai.StreamEventFinish, FinishReason: ai.FinishReasonStop},
		},
	}
	result, err := ai.GenerateText(context.Background(), ai.GenerateTextRequest{
		Model:    model,
		Messages: []ai.Message{ai.UserMessage("hi")},
		MaxSteps: 1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalUsage.PromptTokens != 100 {
		t.Errorf("PromptTokens: got %d", result.TotalUsage.PromptTokens)
	}
	if result.TotalUsage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens: got %d", result.TotalUsage.CompletionTokens)
	}
	if result.TotalUsage.TotalTokens != 150 {
		t.Errorf("TotalTokens: got %d", result.TotalUsage.TotalTokens)
	}
}
