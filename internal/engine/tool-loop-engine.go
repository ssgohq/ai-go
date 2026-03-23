package engine

import (
	"context"
	"fmt"
)

const defaultMaxSteps = 10

// Run executes the tool loop and streams StepEvents onto the returned channel.
// The channel is closed when the run completes or encounters an unrecoverable error.
func Run(ctx context.Context, params RunParams) <-chan StepEvent {
	ch := make(chan StepEvent, 64)
	go runLoop(ctx, ch, params)
	return ch
}

func runLoop(ctx context.Context, out chan<- StepEvent, params RunParams) {
	defer close(out)

	maxSteps := params.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}

	history := buildInitialHistory(params.Request)
	var completedSteps []StepResultInfo

	for step := 0; step < maxSteps; step++ {
		out <- StepEvent{Type: StepEventStepStart, StepNumber: step}

		model := params.Model
		req := params.Request
		req.System = "" // already prepended as system message in history
		req.Messages = history

		if params.PrepareStep != nil {
			psResult := params.PrepareStep(PrepareStepContext{
				StepNumber: step,
				Steps:      completedSteps,
			})
			if psResult != nil {
				if psResult.Model != nil {
					model = psResult.Model
				}
				if psResult.ToolChoice != nil {
					req.ToolChoice = psResult.ToolChoice
				}
				if psResult.System != "" {
					req.System = psResult.System
				}
				if psResult.ProviderOptions != nil {
					req.ProviderOptions = mergeProviderOptions(req.ProviderOptions, psResult.ProviderOptions)
				}
				if psResult.ActiveTools != nil {
					req.Tools = filterTools(req.Tools, psResult.ActiveTools)
				}
			}
		}

		eventCh, err := model.Stream(ctx, req)
		if err != nil {
			out <- StepEvent{Type: StepEventError, Error: fmt.Errorf("step %d: start stream: %w", step, err)}
			return
		}

		acc := newToolCallAccumulator()
		sr, fatalErr := consumeStream(eventCh, out, acc, params.Callbacks)
		if fatalErr {
			return
		}
		fullText := sr.text

		if !acc.hasToolCalls() {
			stepEndEv := StepEvent{
				Type:             StepEventStepEnd,
				StepNumber:       step,
				FinishReason:     sr.finish,
				RawFinishReason:  sr.rawFinish,
				ProviderMetadata: sr.providerMeta,
				Warnings:         sr.warnings,
			}
			out <- stepEndEv
			emitOnStepFinish(params.Callbacks, step, nil, nil, sr)
			completedSteps = append(completedSteps, StepResultInfo{
				StepNumber: step, Text: fullText, FinishReason: sr.finish,
			})
			emitStructuredOutput(ctx, out, params, history)
			emitOnFinish(params.Callbacks, completedSteps, sr)
			out <- StepEvent{Type: StepEventDone}
			return
		}

		toolCalls := acc.completed()
		history = append(history, buildAssistantToolCallMessage(fullText, toolCalls))

		toolNames, stepToolCalls, stepToolResults := executeToolCalls(ctx, out, params.Tools, toolCalls, &history)

		out <- StepEvent{
			Type:             StepEventStepEnd,
			StepNumber:       step,
			FinishReason:     sr.finish,
			RawFinishReason:  sr.rawFinish,
			ProviderMetadata: sr.providerMeta,
			Warnings:         sr.warnings,
		}
		emitOnStepFinish(params.Callbacks, step, stepToolCalls, stepToolResults, sr)

		completedSteps = append(completedSteps, StepResultInfo{
			StepNumber: step, HasToolCalls: true, ToolNames: toolNames,
			Text: fullText, FinishReason: sr.finish,
		})

		if params.StopWhen != nil {
			stopResult := &StepResult{HasToolCalls: true, ToolNames: toolNames, Text: fullText}
			if params.StopWhen(step+1, stopResult) {
				emitStructuredOutput(ctx, out, params, history)
				emitOnFinish(params.Callbacks, completedSteps, sr)
				out <- StepEvent{Type: StepEventDone}
				return
			}
		}
	}

	// maxSteps exhausted after tool calls — run one final generation so the model
	// can produce a text response that incorporates the tool results.
	if ok := emitFinalGeneration(ctx, out, params, history, maxSteps); !ok {
		return
	}

	emitStructuredOutput(ctx, out, params, history)
	emitOnFinish(params.Callbacks, completedSteps, streamResult{finish: FinishReasonStop})
	out <- StepEvent{Type: StepEventDone}
}

// streamResult holds accumulated metadata from consuming a model stream.
type streamResult struct {
	text         string
	reasoning    string
	finish       FinishReason
	rawFinish    string
	providerMeta map[string]any
	warnings     []Warning
	usage        *Usage
}

// consumeStream reads all events from a model stream, forwards them to the step
// event channel, and accumulates tool calls via acc (may be nil for text-only calls).
// Returns the accumulated result and true if a fatal error was emitted.
func consumeStream(
	eventCh <-chan StreamEvent,
	out chan<- StepEvent,
	acc *toolCallAccumulator,
	cb *LifecycleCallbacks,
) (streamResult, bool) {
	var sr streamResult
	for ev := range eventCh {
		if fatal := applyStreamEvent(ev, &sr, acc, out, cb); fatal {
			return sr, true
		}
	}
	return sr, false
}

// applyStreamEvent dispatches a single StreamEvent: updates sr in-place,
// forwards StepEvents to out, and fires lifecycle callbacks. Returns true
// if the event was a fatal StreamEventError.
func applyStreamEvent(
	ev StreamEvent,
	sr *streamResult,
	acc *toolCallAccumulator,
	out chan<- StepEvent,
	cb *LifecycleCallbacks,
) bool {
	emitChunk := func(stepEv StepEvent) {
		out <- stepEv
		if cb != nil && cb.OnChunk != nil {
			cb.OnChunk(stepEv)
		}
	}
	switch ev.Type {
	case StreamEventTextDelta:
		sr.text += ev.TextDelta
		emitChunk(StepEvent{Type: StepEventTextDelta, TextDelta: ev.TextDelta})

	case StreamEventReasoningDelta:
		sr.reasoning += ev.TextDelta
		emitChunk(StepEvent{
			Type:             StepEventReasoningDelta,
			ReasoningDelta:   ev.TextDelta,
			ThoughtSignature: ev.ThoughtSignature,
		})

	case StreamEventToolCallDelta:
		handleToolCallDelta(ev, acc, out, cb)

	case StreamEventUsage:
		sr.usage = ev.Usage
		emitChunk(StepEvent{Type: StepEventUsage, Usage: ev.Usage})

	case StreamEventSource:
		if ev.Source != nil {
			emitChunk(StepEvent{Type: StepEventSource, Source: ev.Source})
		}

	case StreamEventFinish:
		sr.finish = ev.FinishReason
		sr.rawFinish = ev.RawFinishReason
		sr.providerMeta = ev.ProviderMetadata
		if len(ev.Warnings) > 0 {
			sr.warnings = append(sr.warnings, ev.Warnings...)
		}

	case StreamEventError:
		out <- StepEvent{Type: StepEventError, Error: ev.Error}
		if cb != nil && cb.OnError != nil {
			cb.OnError(ev.Error)
		}
		return true
	}
	return false
}

// handleToolCallDelta handles a StreamEventToolCallDelta event by forwarding
// either a tool-call-start or a tool-call-delta StepEvent to out and the
// optional chunk callback. It is a no-op when acc is nil.
func handleToolCallDelta(
	ev StreamEvent,
	acc *toolCallAccumulator,
	out chan<- StepEvent,
	cb *LifecycleCallbacks,
) {
	if acc == nil {
		return
	}
	isNew := acc.add(ev)
	if isNew {
		stepEv := StepEvent{
			Type:              StepEventToolCallStart,
			ToolCallIndex:     ev.ToolCallIndex,
			ToolCallID:        ev.ToolCallID,
			ToolCallName:      ev.ToolCallName,
			ToolCallArgsDelta: ev.ToolCallArgsDelta,
			ThoughtSignature:  ev.ThoughtSignature,
		}
		out <- stepEv
		if cb != nil && cb.OnChunk != nil {
			cb.OnChunk(stepEv)
		}
	} else if ev.ToolCallArgsDelta != "" {
		stepEv := StepEvent{
			Type:              StepEventToolCallDelta,
			ToolCallIndex:     ev.ToolCallIndex,
			ToolCallID:        ev.ToolCallID,
			ToolCallArgsDelta: ev.ToolCallArgsDelta,
		}
		out <- stepEv
		if cb != nil && cb.OnChunk != nil {
			cb.OnChunk(stepEv)
		}
	}
}

// emitFinalGeneration runs a tool-free generation after maxSteps are exhausted so the
// model produces a text response incorporating earlier tool results. Returns false if
// a fatal error was emitted and the caller should return immediately.
func emitFinalGeneration(
	ctx context.Context,
	out chan<- StepEvent,
	params RunParams,
	history []Message,
	stepNum int,
) bool {
	out <- StepEvent{Type: StepEventStepStart, StepNumber: stepNum}

	// Strip tools so the model is forced to produce text, not more tool calls.
	eventCh, err := params.Model.Stream(ctx, Request{Messages: history})
	if err != nil {
		out <- StepEvent{Type: StepEventError, Error: fmt.Errorf("final step: start stream: %w", err)}
		return false
	}

	sr, fatalErr := consumeStream(eventCh, out, nil, params.Callbacks)
	if fatalErr {
		return false
	}

	out <- StepEvent{
		Type:             StepEventStepEnd,
		StepNumber:       stepNum,
		FinishReason:     sr.finish,
		RawFinishReason:  sr.rawFinish,
		ProviderMetadata: sr.providerMeta,
		Warnings:         sr.warnings,
	}
	return true
}

// executeToolCalls processes a batch of completed tool calls: fires events,
// runs executors, appends tool-result messages to history, and returns the
// accumulated names, call infos, and results for step-finish accounting.
func executeToolCalls(
	ctx context.Context,
	out chan<- StepEvent,
	tools *ToolSet,
	toolCalls []toolCallState,
	history *[]Message,
) (toolNames []string, stepToolCalls []ToolCallInfo, stepToolResults []ToolResult) {
	toolNames = make([]string, 0, len(toolCalls))
	for _, tc := range toolCalls {
		out <- StepEvent{Type: StepEventToolCallReady, ToolCallID: tc.id, ToolCallName: tc.name}

		result := executeToolCall(ctx, tools, tc)

		// Apply ToModelOutput transform for history; event keeps original output.
		modelOutput := result.Output
		if tools != nil {
			for _, def := range tools.Definitions {
				if def.Name == tc.name && def.ToModelOutput != nil {
					modelOutput = def.ToModelOutput(result.Output)
					break
				}
			}
		}

		*history = append(*history, buildToolResultMessage(tc.id, tc.name, modelOutput))
		out <- StepEvent{Type: StepEventToolResult, ToolResult: result}
		toolNames = append(toolNames, tc.name)
		stepToolCalls = append(stepToolCalls, ToolCallInfo{ID: tc.id, Name: tc.name, Args: tc.args})
		stepToolResults = append(stepToolResults, *result)
	}
	return toolNames, stepToolCalls, stepToolResults
}

func executeToolCall(ctx context.Context, tools *ToolSet, tc toolCallState) *ToolResult {
	result := &ToolResult{ID: tc.id, Name: tc.name, Args: tc.args}
	if tools == nil || tools.Executor == nil {
		result.Output = fmt.Sprintf(`{"error":"no executor for tool %q"}`, tc.name)
		return result
	}
	// Inject tool call ID into context so downstream code (e.g. approval managers) can correlate.
	execCtx := context.WithValue(ctx, toolCallIDCtxKey, tc.id)
	output, err := tools.Executor.Execute(execCtx, tc.name, tc.args)
	if err != nil {
		result.Output = fmt.Sprintf(`{"error":%q}`, err.Error())
	} else {
		result.Output = output
	}
	return result
}

func buildInitialHistory(req Request) []Message {
	msgs := make([]Message, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, Message{Role: "system", Content: []ContentPart{{Type: "text", Text: req.System}}})
	}
	msgs = append(msgs, req.Messages...)
	return msgs
}

func buildAssistantToolCallMessage(text string, calls []toolCallState) Message {
	parts := make([]ContentPart, 0, 1+len(calls))
	if text != "" {
		parts = append(parts, ContentPart{Type: "text", Text: text})
	}
	for _, tc := range calls {
		parts = append(parts, ContentPart{
			Type:             "tool_call",
			ToolCallID:       tc.id,
			ToolCallName:     tc.name,
			ToolCallArgs:     tc.args,
			ThoughtSignature: tc.thoughtSignature,
		})
	}
	return Message{Role: "assistant", Content: parts}
}

func buildToolResultMessage(toolCallID, toolName, output string) Message {
	return Message{
		Role: "tool",
		Content: []ContentPart{{
			Type:             "tool_result",
			ToolResultID:     toolCallID,
			ToolResultName:   toolName,
			ToolResultOutput: output,
		}},
	}
}

func mergeProviderOptions(base, override map[string]any) map[string]any {
	if base == nil {
		return override
	}
	merged := make(map[string]any, len(base)+len(override))
	for k, v := range base {
		merged[k] = v
	}
	for k, v := range override {
		merged[k] = v
	}
	return merged
}

func filterTools(tools []ToolDefinition, active []string) []ToolDefinition {
	if len(active) == 0 {
		return nil
	}
	set := make(map[string]bool, len(active))
	for _, name := range active {
		set[name] = true
	}
	var filtered []ToolDefinition
	for _, t := range tools {
		if set[t.Name] {
			filtered = append(filtered, t)
		}
	}
	return filtered
}

func emitOnStepFinish(
	cb *LifecycleCallbacks,
	step int,
	toolCalls []ToolCallInfo,
	toolResults []ToolResult,
	sr streamResult,
) {
	if cb == nil || cb.OnStepFinish == nil {
		return
	}
	cb.OnStepFinish(StepFinishEvent{
		StepNumber:       step,
		ToolCalls:        toolCalls,
		ToolResults:      toolResults,
		FinishReason:     sr.finish,
		Usage:            sr.usage,
		ProviderMetadata: sr.providerMeta,
		Warnings:         sr.warnings,
	})
}

func emitOnFinish(cb *LifecycleCallbacks, steps []StepResultInfo, sr streamResult) {
	if cb == nil || cb.OnFinish == nil {
		return
	}
	var totalText, totalReasoning string
	var totalUsage Usage
	var lastFinish FinishReason
	var lastMeta map[string]any
	for _, s := range steps {
		totalText += s.Text
		lastFinish = s.FinishReason
	}
	if sr.usage != nil {
		totalUsage = Usage{
			PromptTokens:     sr.usage.PromptTokens,
			CompletionTokens: sr.usage.CompletionTokens,
			TotalTokens:      sr.usage.TotalTokens,
			ReasoningTokens:  sr.usage.ReasoningTokens,
		}
	}
	totalReasoning = sr.reasoning
	lastMeta = sr.providerMeta
	if sr.finish != "" {
		lastFinish = sr.finish
	}
	cb.OnFinish(FinishEvent{
		Text:             totalText,
		Reasoning:        totalReasoning,
		Steps:            steps,
		TotalUsage:       totalUsage,
		FinishReason:     lastFinish,
		ProviderMetadata: lastMeta,
	})
}
