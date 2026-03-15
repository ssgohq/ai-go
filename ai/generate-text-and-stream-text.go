package ai

import (
	"context"
	"encoding/json"

	"github.com/ssgohq/ai-go/internal/engine"
)

// GenerateText runs a full tool loop and returns the aggregated result.
func GenerateText(ctx context.Context, req GenerateTextRequest) (*GenerateTextResult, error) {
	ch := engine.Run(ctx, toEngineParams(req))

	result := &GenerateTextResult{}
	var currentStep *StepOutput

	for ev := range ch {
		switch ev.Type {
		case engine.StepEventStepStart:
			currentStep = &StepOutput{}

		case engine.StepEventTextDelta:
			result.Text += ev.TextDelta
			if currentStep != nil {
				currentStep.Text += ev.TextDelta
			}

		case engine.StepEventReasoningDelta:
			result.Reasoning += ev.ReasoningDelta
			if currentStep != nil {
				currentStep.Reasoning += ev.ReasoningDelta
			}

		case engine.StepEventToolCallStart:
			if currentStep != nil {
				currentStep.ToolCalls = append(currentStep.ToolCalls, ToolCallOutput{
					ID:   ev.ToolCallID,
					Name: ev.ToolCallName,
				})
			}

		case engine.StepEventToolResult:
			if ev.ToolResult != nil {
				tr := ToolResult{
					ID:     ev.ToolResult.ID,
					Name:   ev.ToolResult.Name,
					Args:   ev.ToolResult.Args,
					Output: ev.ToolResult.Output,
				}
				result.ToolResults = append(result.ToolResults, tr)
				if currentStep != nil {
					currentStep.ToolResults = append(currentStep.ToolResults, tr)
				}
			}

		case engine.StepEventUsage:
			if ev.Usage != nil {
				result.TotalUsage.PromptTokens += ev.Usage.PromptTokens
				result.TotalUsage.CompletionTokens += ev.Usage.CompletionTokens
				result.TotalUsage.TotalTokens += ev.Usage.TotalTokens
				if currentStep != nil {
					currentStep.Usage = Usage(*ev.Usage)
				}
			}

		case engine.StepEventStepEnd:
			if currentStep != nil {
				currentStep.FinishReason = FinishReason(ev.FinishReason)
				result.Steps = append(result.Steps, *currentStep)
				result.FinishReason = FinishReason(ev.FinishReason)
				currentStep = nil
			}

		case engine.StepEventStructuredOutput:
			result.StructuredOutput = ev.StructuredOutput

		case engine.StepEventError:
			return result, ev.Error
		}
	}

	return result, nil
}

// StreamText runs the tool loop and returns a channel of engine.StepEvents for
// callers that need live streaming (e.g. SSE adapters).
func StreamText(ctx context.Context, req GenerateTextRequest) <-chan engine.StepEvent {
	return engine.Run(ctx, toEngineParams(req))
}

// toEngineParams converts a public GenerateTextRequest to engine.RunParams.
// It also wraps the ai.LanguageModel to satisfy engine.Model.
func toEngineParams(req GenerateTextRequest) engine.RunParams {
	engReq := engine.Request{
		System:   req.System,
		Messages: toEngineMessages(req.Messages),
		Settings: engine.CallSettings{
			Temperature:   req.Settings.Temperature,
			MaxTokens:     req.Settings.MaxTokens,
			StopSequences: req.Settings.StopSequences,
		},
	}
	if req.Output != nil {
		engReq.Output = &engine.OutputSchema{Type: req.Output.Type, Schema: req.Output.Schema}
	}

	var engTools *engine.ToolSet
	if req.Tools != nil {
		defs := make([]engine.ToolDefinition, len(req.Tools.Definitions))
		for i, d := range req.Tools.Definitions {
			defs[i] = engine.ToolDefinition{
				Name:        d.Name,
				Description: d.Description,
				Parameters:  d.Parameters,
			}
		}
		engTools = &engine.ToolSet{
			Definitions: defs,
			Executor:    req.Tools.Executor,
		}
		engReq.Tools = defs
	}

	var stopWhen engine.StopCondition
	if req.StopWhen != nil {
		sw := req.StopWhen
		stopWhen = func(step int, r *engine.StepResult) bool {
			return sw(step, &StepResult{
				HasToolCalls: r.HasToolCalls,
				ToolNames:    r.ToolNames,
				Text:         r.Text,
			})
		}
	}

	return engine.RunParams{
		Model:    &engineModelAdapter{req.Model},
		Request:  engReq,
		Tools:    engTools,
		StopWhen: stopWhen,
		MaxSteps: req.MaxSteps,
	}
}

// engineModelAdapter wraps an ai.LanguageModel to satisfy engine.Model.
type engineModelAdapter struct {
	m LanguageModel
}

func (a *engineModelAdapter) ModelID() string { return a.m.ModelID() }

func (a *engineModelAdapter) Stream(ctx context.Context, req engine.Request) (<-chan engine.StreamEvent, error) {
	aiReq := LanguageModelRequest{
		System:   req.System,
		Messages: fromEngineMessages(req.Messages),
		Settings: CallSettings{
			Temperature:   req.Settings.Temperature,
			MaxTokens:     req.Settings.MaxTokens,
			StopSequences: req.Settings.StopSequences,
		},
	}
	if req.Output != nil {
		aiReq.Output = &OutputSchema{Type: req.Output.Type, Schema: req.Output.Schema}
	}
	for _, td := range req.Tools {
		aiReq.Tools = append(aiReq.Tools, ToolDefinition{
			Name:        td.Name,
			Description: td.Description,
			Parameters:  td.Parameters,
		})
	}

	aiCh, err := a.m.Stream(ctx, aiReq)
	if err != nil {
		return nil, err
	}

	engCh := make(chan engine.StreamEvent, 64)
	go func() {
		defer close(engCh)
		for ev := range aiCh {
			engCh <- toEngineStreamEvent(ev)
		}
	}()
	return engCh, nil
}

func toEngineStreamEvent(ev StreamEvent) engine.StreamEvent {
	e := engine.StreamEvent{
		Type:              engine.StreamEventType(ev.Type),
		TextDelta:         ev.TextDelta,
		ToolCallIndex:     ev.ToolCallIndex,
		ToolCallID:        ev.ToolCallID,
		ToolCallName:      ev.ToolCallName,
		ToolCallArgsDelta: ev.ToolCallArgsDelta,
		ThoughtSignature:  ev.ThoughtSignature,
		FinishReason:      engine.FinishReason(ev.FinishReason),
		Error:             ev.Error,
	}
	if ev.Usage != nil {
		e.Usage = &engine.Usage{
			PromptTokens:     ev.Usage.PromptTokens,
			CompletionTokens: ev.Usage.CompletionTokens,
			TotalTokens:      ev.Usage.TotalTokens,
		}
	}
	return e
}

func toEngineMessages(msgs []Message) []engine.Message {
	out := make([]engine.Message, len(msgs))
	for i, m := range msgs {
		out[i] = engine.Message{
			Role:    string(m.Role),
			Content: toEngineContentParts(m.Content),
		}
	}
	return out
}

func toEngineContentParts(parts []ContentPart) []engine.ContentPart {
	out := make([]engine.ContentPart, len(parts))
	for i, p := range parts {
		ep := engine.ContentPart{Type: string(p.Type), Text: p.Text, ImageURL: p.ImageURL}
		ep.ToolCallID = p.ToolCallID
		ep.ToolCallName = p.ToolCallName
		if p.ToolCallArgs != nil {
			ep.ToolCallArgs = string(p.ToolCallArgs)
		}
		ep.ToolResultID = p.ToolResultID
		ep.ToolResultOutput = p.ToolResultOutput
		out[i] = ep
	}
	return out
}

func fromEngineMessages(msgs []engine.Message) []Message {
	out := make([]Message, len(msgs))
	for i, m := range msgs {
		out[i] = Message{
			Role:    Role(m.Role),
			Content: fromEngineContentParts(m.Content),
		}
	}
	return out
}

func fromEngineContentParts(parts []engine.ContentPart) []ContentPart {
	out := make([]ContentPart, len(parts))
	for i, p := range parts {
		cp := ContentPart{
			Type:             ContentPartType(p.Type),
			Text:             p.Text,
			ImageURL:         p.ImageURL,
			ToolCallID:       p.ToolCallID,
			ToolCallName:     p.ToolCallName,
			ToolResultID:     p.ToolResultID,
			ToolResultOutput: p.ToolResultOutput,
		}
		if p.ToolCallArgs != "" {
			cp.ToolCallArgs = json.RawMessage(p.ToolCallArgs)
		}
		out[i] = cp
	}
	return out
}
