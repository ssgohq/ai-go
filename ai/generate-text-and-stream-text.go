package ai

import (
	"context"
	"encoding/json"

	"github.com/open-ai-sdk/ai-go/internal/engine"
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
			currentStep = handleToolResult(ev, result, currentStep)

		case engine.StepEventUsage:
			handleUsage(ev, result, currentStep)

		case engine.StepEventSource:
			handleSource(ev, result, currentStep)

		case engine.StepEventFileDelta:
			handleFileDelta(ev, result, currentStep)

		case engine.StepEventStepEnd:
			currentStep = handleStepEnd(ev, result, currentStep)

		case engine.StepEventStructuredOutput:
			result.StructuredOutput = ev.StructuredOutput

		case engine.StepEventError:
			return result, ev.Error
		}
	}

	return result, nil
}

func handleToolResult(ev engine.StepEvent, result *GenerateTextResult, step *StepOutput) *StepOutput {
	if ev.ToolResult == nil {
		return step
	}
	tr := ToolResult{
		ID:      ev.ToolResult.ID,
		Name:    ev.ToolResult.Name,
		Args:    ev.ToolResult.Args,
		Output:  ev.ToolResult.Output,
		Content: fromEngineToolResultContent(ev.ToolResult.Content),
	}
	result.ToolResults = append(result.ToolResults, tr)
	if step != nil {
		step.ToolResults = append(step.ToolResults, tr)
	}
	return step
}

func handleUsage(ev engine.StepEvent, result *GenerateTextResult, step *StepOutput) {
	if ev.Usage == nil {
		return
	}
	result.TotalUsage.PromptTokens += ev.Usage.PromptTokens
	result.TotalUsage.CompletionTokens += ev.Usage.CompletionTokens
	result.TotalUsage.TotalTokens += ev.Usage.TotalTokens
	result.TotalUsage.ReasoningTokens += ev.Usage.ReasoningTokens
	result.TotalUsage.CacheReadTokens += ev.Usage.CacheReadTokens
	result.TotalUsage.CacheWriteTokens += ev.Usage.CacheWriteTokens
	if step != nil {
		step.Usage = Usage{
			PromptTokens:     ev.Usage.PromptTokens,
			CompletionTokens: ev.Usage.CompletionTokens,
			TotalTokens:      ev.Usage.TotalTokens,
			ReasoningTokens:  ev.Usage.ReasoningTokens,
			CacheReadTokens:  ev.Usage.CacheReadTokens,
			CacheWriteTokens: ev.Usage.CacheWriteTokens,
		}
	}
}

func handleSource(ev engine.StepEvent, result *GenerateTextResult, step *StepOutput) {
	if ev.Source == nil {
		return
	}
	src := Source{
		SourceType:       ev.Source.SourceType,
		ID:               ev.Source.ID,
		URL:              ev.Source.URL,
		Title:            ev.Source.Title,
		ProviderMetadata: ev.Source.ProviderMetadata,
	}
	result.Sources = append(result.Sources, src)
	if step != nil {
		step.Sources = append(step.Sources, src)
	}
}

func handleFileDelta(ev engine.StepEvent, result *GenerateTextResult, step *StepOutput) {
	if len(ev.FileData) == 0 {
		return
	}
	f := GeneratedFile{
		Data:     ev.FileData,
		MimeType: ev.FileMimeType,
	}
	result.Files = append(result.Files, f)
	if step != nil {
		step.Files = append(step.Files, f)
	}
}

func handleStepEnd(ev engine.StepEvent, result *GenerateTextResult, step *StepOutput) *StepOutput {
	if step == nil {
		return nil
	}
	step.FinishReason = FinishReason(ev.FinishReason)
	step.RawFinishReason = ev.RawFinishReason
	step.ProviderMetadata = ev.ProviderMetadata
	step.Warnings = fromEngineWarnings(ev.Warnings)
	result.Steps = append(result.Steps, *step)
	result.FinishReason = FinishReason(ev.FinishReason)
	result.RawFinishReason = ev.RawFinishReason
	result.ProviderMetadata = ev.ProviderMetadata
	result.Warnings = append(result.Warnings, step.Warnings...)
	return nil
}

// StreamText runs the tool loop and returns a *StreamResult for callers that
// need live streaming (e.g. SSE adapters). Use StreamResult.TextStream() for
// text deltas, StreamResult.Events() for raw engine events, or
// StreamResult.Consume() to block and get the full aggregated result.
func StreamText(ctx context.Context, req GenerateTextRequest) *StreamResult {
	ch := engine.Run(ctx, toEngineParams(req))
	if req.SmoothStream != nil {
		ch = req.SmoothStream.Transform(ctx, ch)
	}
	return NewStreamResult(ch)
}

// toEngineParams converts a public GenerateTextRequest to engine.RunParams.
// It also wraps the ai.LanguageModel to satisfy engine.Model.
func toEngineParams(req GenerateTextRequest) engine.RunParams {
	engReq := engine.Request{
		System:          req.System,
		Messages:        toEngineMessages(req.Messages),
		ProviderOptions: req.ProviderOptions,
		Settings: engine.CallSettings{
			Temperature:   req.Settings.Temperature,
			MaxTokens:     req.Settings.MaxTokens,
			TopP:          req.Settings.TopP,
			TopK:          req.Settings.TopK,
			Seed:          req.Settings.Seed,
			StopSequences: req.Settings.StopSequences,
		},
	}
	if req.Output != nil {
		engReq.Output = &engine.OutputSchema{Type: req.Output.Type, Schema: req.Output.Schema}
	}
	if req.ToolChoice != nil {
		engReq.ToolChoice = &engine.ToolChoice{Type: req.ToolChoice.Type, ToolName: req.ToolChoice.ToolName}
	}

	var engTools *engine.ToolSet
	if req.Tools != nil {
		defs := make([]engine.ToolDefinition, len(req.Tools.Definitions))
		for i, d := range req.Tools.Definitions {
			defs[i] = engine.ToolDefinition{
				Name:          d.Name,
				Description:   d.Description,
				InputSchema:   d.InputSchema,
				ToModelOutput: d.ToModelOutput,
			}
		}
		engTools = &engine.ToolSet{
			Definitions: defs,
			Executor:    req.Tools.Executor,
		}
		engReq.Tools = defs

		if len(req.ActiveTools) > 0 {
			engReq.Tools = engineFilterTools(defs, req.ActiveTools)
		}
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

	var engPrepareStep engine.PrepareStepFunc
	if req.PrepareStep != nil {
		engPrepareStep = func(ectx engine.PrepareStepContext) *engine.PrepareStepResult {
			aiCtx := PrepareStepContext{StepNumber: ectx.StepNumber}
			for _, s := range ectx.Steps {
				aiCtx.Steps = append(aiCtx.Steps, PrepareStepInfo{
					StepNumber:   s.StepNumber,
					HasToolCalls: s.HasToolCalls,
					ToolNames:    s.ToolNames,
					Text:         s.Text,
					FinishReason: FinishReason(s.FinishReason),
				})
			}
			result := req.PrepareStep(aiCtx)
			if result == nil {
				return nil
			}
			engResult := &engine.PrepareStepResult{
				ActiveTools:     result.ActiveTools,
				System:          result.System,
				ProviderOptions: result.ProviderOptions,
			}
			if result.Model != nil {
				engResult.Model = &engineModelAdapter{result.Model}
			}
			if result.ToolChoice != nil {
				engResult.ToolChoice = &engine.ToolChoice{
					Type:     result.ToolChoice.Type,
					ToolName: result.ToolChoice.ToolName,
				}
			}
			return engResult
		}
	}

	var engCallbacks *engine.LifecycleCallbacks
	if req.OnStepFinish != nil || req.OnFinish != nil || req.OnChunk != nil || req.OnError != nil {
		engCallbacks = &engine.LifecycleCallbacks{}
		if req.OnStepFinish != nil {
			fn := req.OnStepFinish
			engCallbacks.OnStepFinish = func(ev engine.StepFinishEvent) {
				fn(StepFinishEvent{
					StepNumber:       ev.StepNumber,
					ToolCalls:        fromEngineToolCalls(ev.ToolCalls),
					ToolResults:      fromEngineToolResults(ev.ToolResults),
					FinishReason:     FinishReason(ev.FinishReason),
					Usage:            fromEngineUsagePtr(ev.Usage),
					ProviderMetadata: ev.ProviderMetadata,
					Warnings:         fromEngineWarnings(ev.Warnings),
				})
			}
		}
		if req.OnFinish != nil {
			fn := req.OnFinish
			engCallbacks.OnFinish = func(ev engine.FinishEvent) {
				fn(FinishEvent{
					Text:             ev.Text,
					Reasoning:        ev.Reasoning,
					Steps:            fromEngineStepInfos(ev.Steps),
					TotalUsage:       fromEngineUsage(ev.TotalUsage),
					FinishReason:     FinishReason(ev.FinishReason),
					ProviderMetadata: ev.ProviderMetadata,
				})
			}
		}
		if req.OnChunk != nil {
			fn := req.OnChunk
			engCallbacks.OnChunk = func(ev engine.StepEvent) {
				fn(toChunkEvent(ev))
			}
		}
		if req.OnError != nil {
			engCallbacks.OnError = req.OnError
		}
	}

	return engine.RunParams{
		Model:                 &engineModelAdapter{req.Model},
		Request:               engReq,
		Tools:                 engTools,
		StopWhen:              stopWhen,
		MaxSteps:              req.MaxSteps,
		PrepareStep:           engPrepareStep,
		Callbacks:             engCallbacks,
		ParallelToolExecution: req.ParallelToolExecution,
		MaxParallelTools:      req.MaxParallelTools,
	}
}

func engineFilterTools(tools []engine.ToolDefinition, active []string) []engine.ToolDefinition {
	set := make(map[string]bool, len(active))
	for _, name := range active {
		set[name] = true
	}
	var filtered []engine.ToolDefinition
	for _, t := range tools {
		if set[t.Name] {
			filtered = append(filtered, t)
		}
	}
	return filtered
}

func fromEngineToolCalls(tcs []engine.ToolCallInfo) []ToolCallOutput {
	if len(tcs) == 0 {
		return nil
	}
	out := make([]ToolCallOutput, len(tcs))
	for i, tc := range tcs {
		out[i] = ToolCallOutput{ID: tc.ID, Name: tc.Name, Args: json.RawMessage(tc.Args)}
	}
	return out
}

func fromEngineToolResults(trs []engine.ToolResult) []ToolResult {
	if len(trs) == 0 {
		return nil
	}
	out := make([]ToolResult, len(trs))
	for i, tr := range trs {
		out[i] = ToolResult{
			ID: tr.ID, Name: tr.Name, Args: tr.Args,
			Output:  tr.Output,
			Content: fromEngineToolResultContent(tr.Content),
		}
	}
	return out
}

func fromEngineToolResultContent(cs []engine.ToolResultContent) []ToolResultContent {
	if len(cs) == 0 {
		return nil
	}
	out := make([]ToolResultContent, len(cs))
	for i, c := range cs {
		out[i] = ToolResultContent{Type: c.Type, Text: c.Text, Data: c.Data, MimeType: c.MimeType}
	}
	return out
}

func fromEngineUsagePtr(u *engine.Usage) *Usage {
	if u == nil {
		return nil
	}
	return &Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		ReasoningTokens:  u.ReasoningTokens,
		CacheReadTokens:  u.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens,
	}
}

func fromEngineUsage(u engine.Usage) Usage {
	return Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		ReasoningTokens:  u.ReasoningTokens,
		CacheReadTokens:  u.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens,
	}
}

func fromEngineStepInfos(steps []engine.StepResultInfo) []StepOutput {
	if len(steps) == 0 {
		return nil
	}
	out := make([]StepOutput, len(steps))
	for i, s := range steps {
		out[i] = StepOutput{
			Text:         s.Text,
			FinishReason: FinishReason(s.FinishReason),
		}
	}
	return out
}

func toChunkEvent(ev engine.StepEvent) ChunkEvent {
	var typ string
	switch ev.Type {
	case engine.StepEventTextDelta:
		typ = "text-delta"
	case engine.StepEventReasoningDelta:
		typ = "reasoning-delta"
	case engine.StepEventToolCallStart:
		typ = "tool-call-start"
	case engine.StepEventToolCallDelta:
		typ = "tool-call-delta"
	case engine.StepEventToolCallReady:
		typ = "tool-call-ready"
	case engine.StepEventToolResult:
		typ = "tool-result"
	case engine.StepEventUsage:
		typ = "usage"
	case engine.StepEventStepStart:
		typ = "step-start"
	case engine.StepEventStepEnd:
		typ = "step-end"
	case engine.StepEventDone:
		typ = "done"
	case engine.StepEventError:
		typ = "error"
	case engine.StepEventSource:
		typ = "source"
	case engine.StepEventFileDelta:
		typ = "file-delta"
	default:
		typ = "unknown"
	}
	return ChunkEvent{
		Type:              typ,
		TextDelta:         ev.TextDelta,
		ReasoningDelta:    ev.ReasoningDelta,
		ToolCallID:        ev.ToolCallID,
		ToolCallName:      ev.ToolCallName,
		ToolCallArgsDelta: ev.ToolCallArgsDelta,
		StepNumber:        ev.StepNumber,
		FinishReason:      FinishReason(ev.FinishReason),
	}
}

// engineModelAdapter wraps an ai.LanguageModel to satisfy engine.Model.
type engineModelAdapter struct {
	m LanguageModel
}

func (a *engineModelAdapter) ModelID() string { return a.m.ModelID() }

func (a *engineModelAdapter) Stream(ctx context.Context, req engine.Request) (<-chan engine.StreamEvent, error) {
	aiReq := LanguageModelRequest{
		System:          req.System,
		Messages:        fromEngineMessages(req.Messages),
		ProviderOptions: req.ProviderOptions,
		Settings: CallSettings{
			Temperature:   req.Settings.Temperature,
			MaxTokens:     req.Settings.MaxTokens,
			TopP:          req.Settings.TopP,
			TopK:          req.Settings.TopK,
			Seed:          req.Settings.Seed,
			StopSequences: req.Settings.StopSequences,
		},
	}
	if req.Output != nil {
		aiReq.Output = &OutputSchema{Type: req.Output.Type, Schema: req.Output.Schema}
	}
	if req.ToolChoice != nil {
		aiReq.ToolChoice = &ToolChoice{Type: req.ToolChoice.Type, ToolName: req.ToolChoice.ToolName}
	}
	for _, td := range req.Tools {
		aiReq.Tools = append(aiReq.Tools, ToolDefinition{
			Name:        td.Name,
			Description: td.Description,
			InputSchema: td.InputSchema,
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
		RawFinishReason:   ev.RawFinishReason,
		ProviderMetadata:  ev.ProviderMetadata,
		FileData:          ev.FileData,
		FileMimeType:      ev.FileMimeType,
		Error:             ev.Error,
	}
	if ev.Usage != nil {
		e.Usage = &engine.Usage{
			PromptTokens:     ev.Usage.PromptTokens,
			CompletionTokens: ev.Usage.CompletionTokens,
			TotalTokens:      ev.Usage.TotalTokens,
			ReasoningTokens:  ev.Usage.ReasoningTokens,
			CacheReadTokens:  ev.Usage.CacheReadTokens,
			CacheWriteTokens: ev.Usage.CacheWriteTokens,
		}
	}
	if len(ev.Warnings) > 0 {
		e.Warnings = make([]engine.Warning, len(ev.Warnings))
		for i, w := range ev.Warnings {
			e.Warnings[i] = engine.Warning{Type: w.Type, Message: w.Message, Setting: w.Setting}
		}
	}
	if ev.Source != nil {
		e.Source = &engine.Source{
			SourceType:       ev.Source.SourceType,
			ID:               ev.Source.ID,
			URL:              ev.Source.URL,
			Title:            ev.Source.Title,
			ProviderMetadata: ev.Source.ProviderMetadata,
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
		ep := engine.ContentPart{
			Type:             string(p.Type),
			ImageURL:         p.ImageURL,
			FileURL:          p.FileURL,
			MimeType:         p.MimeType,
			Data:             p.Data,
			FileID:           p.FileID,
			Filename:         p.Filename,
			ToolCallID:       p.ToolCallID,
			ToolCallName:     p.ToolCallName,
			ToolResultID:     p.ToolResultID,
			ToolResultName:   p.ToolResultName,
			ToolResultOutput: p.ToolResultOutput,
		}
		// "reasoning" parts store their text in ReasoningText; map to the engine's Text field.
		if p.Type == ContentPartTypeReasoning {
			ep.Text = p.ReasoningText
		} else {
			ep.Text = p.Text
		}
		if p.ToolCallArgs != nil {
			ep.ToolCallArgs = string(p.ToolCallArgs)
		}
		ep.ThoughtSignature = p.ThoughtSignature
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

func fromEngineWarnings(ws []engine.Warning) []Warning {
	if len(ws) == 0 {
		return nil
	}
	out := make([]Warning, len(ws))
	for i, w := range ws {
		out[i] = Warning{Type: w.Type, Message: w.Message, Setting: w.Setting}
	}
	return out
}

func fromEngineContentParts(parts []engine.ContentPart) []ContentPart {
	out := make([]ContentPart, len(parts))
	for i, p := range parts {
		cp := ContentPart{
			Type:             ContentPartType(p.Type),
			ImageURL:         p.ImageURL,
			FileURL:          p.FileURL,
			MimeType:         p.MimeType,
			Data:             p.Data,
			FileID:           p.FileID,
			Filename:         p.Filename,
			ToolCallID:       p.ToolCallID,
			ToolCallName:     p.ToolCallName,
			ToolResultID:     p.ToolResultID,
			ToolResultName:   p.ToolResultName,
			ToolResultOutput: p.ToolResultOutput,
		}
		// "reasoning" parts map the engine's Text field back to ReasoningText.
		if p.Type == "reasoning" {
			cp.ReasoningText = p.Text
		} else {
			cp.Text = p.Text
		}
		if p.ToolCallArgs != "" {
			cp.ToolCallArgs = json.RawMessage(p.ToolCallArgs)
		}
		cp.ThoughtSignature = p.ThoughtSignature
		out[i] = cp
	}
	return out
}
