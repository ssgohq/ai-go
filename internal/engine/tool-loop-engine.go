package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
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
	// lastSR captures the final iteration's streamResult so we can report an
	// accurate finish reason when the loop exits with pending tool_calls at
	// maxSteps (matching ai-sdk-node: honest return, no forced text step).
	var lastSR streamResult

	for step := 0; step < maxSteps; step++ {
		if ctx.Err() != nil {
			out <- StepEvent{Type: StepEventError, Error: ctx.Err()}
			return
		}

		out <- StepEvent{Type: StepEventStepStart, StepNumber: step}

		model := params.Model
		req := params.Request
		req.System = "" // already prepended as system message in history
		req.Messages = history
		if len(req.Tools) == 0 && params.Tools != nil && len(params.Tools.Definitions) > 0 {
			req.Tools = params.Tools.Definitions
		}

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
		lastSR = sr
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
				StepNumber:       step,
				Text:             fullText,
				Reasoning:        sr.reasoning,
				Usage:            sr.usage,
				FinishReason:     sr.finish,
				RawFinishReason:  sr.rawFinish,
				ProviderMetadata: sr.providerMeta,
				Warnings:         sr.warnings,
			})
			emitStructuredOutput(ctx, out, params, history)
			emitOnFinish(params.Callbacks, completedSteps, sr)
			out <- StepEvent{Type: StepEventDone}
			return
		}

		toolCalls := acc.completed()
		history = append(history, buildAssistantToolCallMessage(fullText, sr.reasoning, toolCalls))

		var toolNames []string
		var stepToolCalls []ToolCallInfo
		var stepToolResults []ToolResult
		if params.ParallelToolExecution {
			toolNames, stepToolCalls, stepToolResults = executeToolCallsParallel(
				ctx, out, params.Tools, params.RepairToolCall, req, toolCalls, &history, params.MaxParallelTools,
			)
		} else {
			toolNames, stepToolCalls, stepToolResults = executeToolCalls(
				ctx, out, params.Tools, params.RepairToolCall, req, toolCalls, &history,
			)
		}

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
			StepNumber:       step,
			HasToolCalls:     true,
			ToolNames:        toolNames,
			Text:             fullText,
			Reasoning:        sr.reasoning,
			ToolCalls:        stepToolCalls,
			ToolResults:      stepToolResults,
			Usage:            sr.usage,
			FinishReason:     sr.finish,
			RawFinishReason:  sr.rawFinish,
			ProviderMetadata: sr.providerMeta,
			Warnings:         sr.warnings,
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

	// maxSteps exhausted with pending tool_calls — exit honestly using the
	// last step's streamResult. Caller sees FinishReasonToolCalls and can
	// decide whether to continue (e.g. bump the budget, force tool_choice=none
	// on a follow-up call) or surface the partial result.
	// Historical note: this used to fire a tool-less "final generation" pass
	// which caused gateway Harmony-parsing issues on gpt-oss/gpt-5 family.
	// Matches ai-sdk-node semantics (see packages/ai generate-text.ts:1008).
	emitStructuredOutput(ctx, out, params, history)
	emitOnFinish(params.Callbacks, completedSteps, lastSR)
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

	case StreamEventFileDelta:
		emitChunk(StepEvent{
			Type:         StepEventFileDelta,
			FileData:     ev.FileData,
			FileMimeType: ev.FileMimeType,
		})

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

// executeToolCalls processes a batch of completed tool calls: validates JSON args,
// emits StepEventToolCallInvalid for invalid args (with error result for model retry),
// then fires events and runs executors for valid calls.
func executeToolCalls(
	ctx context.Context,
	out chan<- StepEvent,
	tools *ToolSet,
	repair ToolCallRepairFunc,
	req Request,
	toolCalls []toolCallState,
	history *[]Message,
) (toolNames []string, stepToolCalls []ToolCallInfo, stepToolResults []ToolResult) {
	prepared := prepareToolCalls(ctx, tools, repair, req, toolCalls)
	toolNames = make([]string, 0, len(toolCalls))
	for _, preparedCall := range prepared {
		tc := preparedCall.tc
		if preparedCall.invalidErr != nil {
			out <- StepEvent{
				Type:              StepEventToolCallInvalid,
				ToolCallID:        tc.id,
				ToolCallName:      tc.name,
				ToolCallArgsDelta: tc.args,
			}
			errOutput := invalidToolCallOutput(tc, preparedCall.invalidErr)
			*history = append(*history, buildToolResultMessage(tc.id, tc.name, errOutput))
			toolNames = append(toolNames, tc.name)
			continue
		}

		out <- StepEvent{
			Type:              StepEventToolCallReady,
			ToolCallID:        tc.id,
			ToolCallName:      tc.name,
			ToolCallArgsDelta: tc.args,
			ThoughtSignature:  tc.thoughtSignature,
		}

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
		stepToolCalls = append(stepToolCalls, ToolCallInfo{
			ID:               tc.id,
			Name:             tc.name,
			Args:             tc.args,
			ThoughtSignature: tc.thoughtSignature,
		})
		stepToolResults = append(stepToolResults, *result)
	}
	return toolNames, stepToolCalls, stepToolResults
}

// executeToolCallsParallel processes tool calls concurrently with a semaphore.
func executeToolCallsParallel(
	ctx context.Context,
	out chan<- StepEvent,
	tools *ToolSet,
	repair ToolCallRepairFunc,
	req Request,
	toolCalls []toolCallState,
	history *[]Message,
	maxParallel int,
) (toolNames []string, stepToolCalls []ToolCallInfo, stepToolResults []ToolResult) {
	if maxParallel <= 0 {
		maxParallel = 5
	}

	type indexedResult struct {
		idx         int
		tc          toolCallState
		result      *ToolResult
		modelOutput string
		valid       bool
		invalidErr  error
	}

	prepared := prepareToolCalls(ctx, tools, repair, req, toolCalls)
	results := make([]indexedResult, len(prepared))
	sem := make(chan struct{}, maxParallel)
	var wg sync.WaitGroup

	for i, preparedCall := range prepared {
		tc := preparedCall.tc
		if preparedCall.invalidErr != nil {
			results[i] = indexedResult{
				idx: i, tc: tc, valid: false, invalidErr: preparedCall.invalidErr,
				result: &ToolResult{
					ID: tc.id, Name: tc.name, Args: tc.args,
					Output: invalidToolCallOutput(tc, preparedCall.invalidErr),
				},
			}
			continue
		}

		// Emit ToolCallReady before execution starts (matches sequential contract)
		out <- StepEvent{
			Type:              StepEventToolCallReady,
			ToolCallID:        tc.id,
			ToolCallName:      tc.name,
			ToolCallArgsDelta: tc.args,
			ThoughtSignature:  tc.thoughtSignature,
		}

		results[i] = indexedResult{idx: i, tc: tc, valid: true}
		wg.Add(1)
		go func(idx int, tc toolCallState) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			result := executeToolCall(ctx, tools, tc)
			modelOutput := result.Output
			if tools != nil {
				for _, def := range tools.Definitions {
					if def.Name == tc.name && def.ToModelOutput != nil {
						modelOutput = def.ToModelOutput(result.Output)
						break
					}
				}
			}
			results[idx].result = result
			results[idx].modelOutput = modelOutput
		}(i, tc)
	}
	wg.Wait()

	// Emit events and build history in original order
	toolNames = make([]string, 0, len(toolCalls))
	for _, r := range results {
		if !r.valid {
			out <- StepEvent{
				Type:              StepEventToolCallInvalid,
				ToolCallID:        r.tc.id,
				ToolCallName:      r.tc.name,
				ToolCallArgsDelta: r.tc.args,
			}
			*history = append(*history, buildToolResultMessage(r.tc.id, r.tc.name, r.result.Output))
			toolNames = append(toolNames, r.tc.name)
			continue
		}

		out <- StepEvent{Type: StepEventToolResult, ToolResult: r.result}
		*history = append(*history, buildToolResultMessage(r.tc.id, r.tc.name, r.modelOutput))
		toolNames = append(toolNames, r.tc.name)
		stepToolCalls = append(stepToolCalls, ToolCallInfo{
			ID:               r.tc.id,
			Name:             r.tc.name,
			Args:             r.tc.args,
			ThoughtSignature: r.tc.thoughtSignature,
		})
		stepToolResults = append(stepToolResults, *r.result)
	}
	return toolNames, stepToolCalls, stepToolResults
}

type preparedToolCall struct {
	tc         toolCallState
	invalidErr error
}

func prepareToolCalls(
	ctx context.Context,
	tools *ToolSet,
	repair ToolCallRepairFunc,
	req Request,
	toolCalls []toolCallState,
) []preparedToolCall {
	stepTools := toolSetForStep(tools, req.Tools)
	prepared := make([]preparedToolCall, 0, len(toolCalls))
	for _, tc := range toolCalls {
		fixed, err := validateAndRepairToolCall(ctx, stepTools, repair, req, tc)
		prepared = append(prepared, preparedToolCall{
			tc:         fixed,
			invalidErr: err,
		})
	}
	return prepared
}

func validateAndRepairToolCall(
	ctx context.Context,
	tools *ToolSet,
	repair ToolCallRepairFunc,
	req Request,
	tc toolCallState,
) (toolCallState, error) {
	err := validateToolCall(tools, tc)
	if err == nil || repair == nil {
		return tc, err
	}

	repaired, repairErr := repair(ctx, ToolCallRepairContext{
		System:   req.System,
		Messages: req.Messages,
		ToolCall: ToolCallInfo{
			ID:               tc.id,
			Name:             tc.name,
			Args:             tc.args,
			ThoughtSignature: tc.thoughtSignature,
		},
		Tools: tools,
		Error: err,
	})
	if repairErr != nil {
		return tc, repairErr
	}
	if repaired == nil {
		return tc, err
	}

	if repaired.ID != "" {
		tc.id = repaired.ID
	}
	if repaired.Name != "" {
		tc.name = repaired.Name
	}
	if repaired.Args != "" || repaired.Args == "" && repaired.Args != tc.args {
		tc.args = repaired.Args
	}
	if repaired.ThoughtSignature != "" {
		tc.thoughtSignature = repaired.ThoughtSignature
	}

	return tc, validateToolCall(tools, tc)
}

func validateToolCall(tools *ToolSet, tc toolCallState) error {
	if tools == nil {
		return &NoSuchToolError{
			ToolName:       tc.name,
			AvailableTools: nil,
		}
	}
	if len(tools.Definitions) == 0 {
		if tc.args != "" && !json.Valid([]byte(tc.args)) {
			return &InvalidToolArgumentsError{
				ToolName: tc.name,
				Args:     tc.args,
			}
		}
		return nil
	}
	if _, ok := findToolDefinition(tools, tc.name); !ok {
		return &NoSuchToolError{
			ToolName:       tc.name,
			AvailableTools: toolDefinitionNames(tools),
		}
	}
	if tc.args != "" && !json.Valid([]byte(tc.args)) {
		return &InvalidToolArgumentsError{
			ToolName: tc.name,
			Args:     tc.args,
		}
	}
	return nil
}

func invalidToolCallOutput(tc toolCallState, err error) string {
	var noSuchToolErr *NoSuchToolError
	if errors.As(err, &noSuchToolErr) {
		return fmt.Sprintf(`{"error":"unknown tool %q"}`, noSuchToolErr.ToolName)
	}

	var invalidArgsErr *InvalidToolArgumentsError
	if errors.As(err, &invalidArgsErr) {
		return fmt.Sprintf(`{"error":"invalid JSON arguments for tool %q"}`, invalidArgsErr.ToolName)
	}

	return fmt.Sprintf(`{"error":%q}`, err.Error())
}

func findToolDefinition(tools *ToolSet, name string) (*ToolDefinition, bool) {
	if tools == nil {
		return nil, false
	}
	for i := range tools.Definitions {
		if tools.Definitions[i].Name == name {
			return &tools.Definitions[i], true
		}
	}
	return nil, false
}

func toolDefinitionNames(tools *ToolSet) []string {
	if tools == nil || len(tools.Definitions) == 0 {
		return nil
	}
	names := make([]string, 0, len(tools.Definitions))
	for _, def := range tools.Definitions {
		names = append(names, def.Name)
	}
	return names
}

func toolSetForStep(tools *ToolSet, activeDefs []ToolDefinition) *ToolSet {
	if tools == nil {
		return nil
	}
	if len(tools.Definitions) == 0 {
		return &ToolSet{Executor: tools.Executor}
	}
	if len(activeDefs) == 0 {
		return nil
	}
	return &ToolSet{
		Definitions: activeDefs,
		Executor:    tools.Executor,
	}
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

func buildAssistantToolCallMessage(text, reasoning string, calls []toolCallState) Message {
	parts := make([]ContentPart, 0, 2+len(calls))
	if reasoning != "" {
		parts = append(parts, ContentPart{Type: "reasoning", Text: reasoning})
	}
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
		Text:             sr.text,
		Reasoning:        sr.reasoning,
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
		totalReasoning += s.Reasoning
		lastFinish = s.FinishReason
		if s.Usage != nil {
			totalUsage.PromptTokens += s.Usage.PromptTokens
			totalUsage.CompletionTokens += s.Usage.CompletionTokens
			totalUsage.TotalTokens += s.Usage.TotalTokens
			totalUsage.ReasoningTokens += s.Usage.ReasoningTokens
			totalUsage.CacheReadTokens += s.Usage.CacheReadTokens
			totalUsage.CacheWriteTokens += s.Usage.CacheWriteTokens
		}
		if s.ProviderMetadata != nil {
			lastMeta = s.ProviderMetadata
		}
	}
	if sr.finish != "" {
		lastFinish = sr.finish
	}
	if sr.providerMeta != nil {
		lastMeta = sr.providerMeta
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
