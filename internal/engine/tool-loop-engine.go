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

	for step := 0; step < maxSteps; step++ {
		out <- StepEvent{Type: StepEventStepStart, StepNumber: step}

		req := params.Request
		req.System = "" // already prepended as system message in history
		req.Messages = history

		eventCh, err := params.Model.Stream(ctx, req)
		if err != nil {
			out <- StepEvent{Type: StepEventError, Error: fmt.Errorf("step %d: start stream: %w", step, err)}
			return
		}

		acc := newToolCallAccumulator()
		sr, fatalErr := consumeStream(eventCh, out, acc)
		if fatalErr {
			return
		}
		fullText := sr.text

		if !acc.hasToolCalls() {
			out <- StepEvent{
				Type:             StepEventStepEnd,
				StepNumber:       step,
				FinishReason:     sr.finish,
				RawFinishReason:  sr.rawFinish,
				ProviderMetadata: sr.providerMeta,
				Warnings:         sr.warnings,
			}
			emitStructuredOutput(ctx, out, params, history)
			out <- StepEvent{Type: StepEventDone}
			return
		}

		toolCalls := acc.completed()
		history = append(history, buildAssistantToolCallMessage(fullText, toolCalls))

		toolNames := make([]string, 0, len(toolCalls))
		for _, tc := range toolCalls {
			out <- StepEvent{Type: StepEventToolCallReady, ToolCallID: tc.id, ToolCallName: tc.name}

			result := executeToolCall(ctx, params.Tools, tc)
			history = append(history, buildToolResultMessage(tc.id, result.Output))
			out <- StepEvent{Type: StepEventToolResult, ToolResult: result}
			toolNames = append(toolNames, tc.name)
		}

		out <- StepEvent{
			Type:             StepEventStepEnd,
			StepNumber:       step,
			FinishReason:     sr.finish,
			RawFinishReason:  sr.rawFinish,
			ProviderMetadata: sr.providerMeta,
			Warnings:         sr.warnings,
		}

		if params.StopWhen != nil {
			stopResult := &StepResult{HasToolCalls: true, ToolNames: toolNames, Text: fullText}
			if params.StopWhen(step+1, stopResult) {
				emitStructuredOutput(ctx, out, params, history)
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
	out <- StepEvent{Type: StepEventDone}
}

// streamResult holds accumulated metadata from consuming a model stream.
type streamResult struct {
	text         string
	finish       FinishReason
	rawFinish    string
	providerMeta map[string]any
	warnings     []Warning
}

// consumeStream reads all events from a model stream, forwards them to the step
// event channel, and accumulates tool calls via acc (may be nil for text-only calls).
// Returns the accumulated result and true if a fatal error was emitted.
func consumeStream(eventCh <-chan StreamEvent, out chan<- StepEvent, acc *toolCallAccumulator) (streamResult, bool) {
	var sr streamResult
	for ev := range eventCh {
		switch ev.Type {
		case StreamEventTextDelta:
			sr.text += ev.TextDelta
			out <- StepEvent{Type: StepEventTextDelta, TextDelta: ev.TextDelta}

		case StreamEventReasoningDelta:
			out <- StepEvent{Type: StepEventReasoningDelta, ReasoningDelta: ev.TextDelta, ThoughtSignature: ev.ThoughtSignature}

		case StreamEventToolCallDelta:
			if acc != nil {
				isNew := acc.add(ev)
				if isNew {
					out <- StepEvent{
						Type:              StepEventToolCallStart,
						ToolCallIndex:     ev.ToolCallIndex,
						ToolCallID:        ev.ToolCallID,
						ToolCallName:      ev.ToolCallName,
						ToolCallArgsDelta: ev.ToolCallArgsDelta,
						ThoughtSignature:  ev.ThoughtSignature,
					}
				} else if ev.ToolCallArgsDelta != "" {
					out <- StepEvent{
						Type:              StepEventToolCallDelta,
						ToolCallIndex:     ev.ToolCallIndex,
						ToolCallID:        ev.ToolCallID,
						ToolCallArgsDelta: ev.ToolCallArgsDelta,
					}
				}
			}

		case StreamEventUsage:
			out <- StepEvent{Type: StepEventUsage, Usage: ev.Usage}

		case StreamEventSource:
			if ev.Source != nil {
				out <- StepEvent{Type: StepEventSource, Source: ev.Source}
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
			return sr, true
		}
	}
	return sr, false
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

	sr, fatalErr := consumeStream(eventCh, out, nil)
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

func executeToolCall(ctx context.Context, tools *ToolSet, tc toolCallState) *ToolResult {
	result := &ToolResult{ID: tc.id, Name: tc.name, Args: tc.args}
	if tools == nil || tools.Executor == nil {
		result.Output = fmt.Sprintf(`{"error":"no executor for tool %q"}`, tc.name)
		return result
	}
	output, err := tools.Executor.Execute(ctx, tc.name, tc.args)
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

func buildToolResultMessage(toolCallID, output string) Message {
	return Message{
		Role: "tool",
		Content: []ContentPart{{
			Type:             "tool_result",
			ToolResultID:     toolCallID,
			ToolResultOutput: output,
		}},
	}
}
