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

		var fullText string
		var lastFinish FinishReason
		acc := newToolCallAccumulator()

		for ev := range eventCh {
			switch ev.Type {
			case StreamEventTextDelta:
				fullText += ev.TextDelta
				out <- StepEvent{Type: StepEventTextDelta, TextDelta: ev.TextDelta}

			case StreamEventReasoningDelta:
				out <- StepEvent{Type: StepEventReasoningDelta, ReasoningDelta: ev.TextDelta}

			case StreamEventToolCallDelta:
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

			case StreamEventUsage:
				out <- StepEvent{Type: StepEventUsage, Usage: ev.Usage}

			case StreamEventFinish:
				lastFinish = ev.FinishReason

			case StreamEventError:
				out <- StepEvent{Type: StepEventError, Error: ev.Error}
				return
			}
		}

		if !acc.hasToolCalls() {
			out <- StepEvent{Type: StepEventStepEnd, StepNumber: step, FinishReason: lastFinish}
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

		out <- StepEvent{Type: StepEventStepEnd, StepNumber: step, FinishReason: lastFinish}

		if params.StopWhen != nil {
			sr := &StepResult{HasToolCalls: true, ToolNames: toolNames, Text: fullText}
			if params.StopWhen(step+1, sr) {
				emitStructuredOutput(ctx, out, params, history)
				out <- StepEvent{Type: StepEventDone}
				return
			}
		}
	}

	emitStructuredOutput(ctx, out, params, history)
	out <- StepEvent{Type: StepEventDone}
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
			Type:         "tool_call",
			ToolCallID:   tc.id,
			ToolCallName: tc.name,
			ToolCallArgs: tc.args,
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
