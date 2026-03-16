package ai

import "github.com/open-ai-sdk/ai-go/internal/engine"

// StreamResult wraps a streaming response with convenient accessors.
// It fans out the source engine channel to text-delta and raw-event subscribers.
type StreamResult struct {
	src <-chan engine.StepEvent

	textCh    chan string
	eventsCh  chan engine.StepEvent
	consumeCh chan engine.StepEvent

	// done is closed when the fan-out goroutine has finished.
	done chan struct{}
}

// NewStreamResult wraps an engine step-event channel in a StreamResult.
func NewStreamResult(ch <-chan engine.StepEvent) *StreamResult {
	sr := &StreamResult{
		src:       ch,
		textCh:    make(chan string, 64),
		eventsCh:  make(chan engine.StepEvent, 64),
		consumeCh: make(chan engine.StepEvent, 64),
		done:      make(chan struct{}),
	}
	sr.start()
	return sr
}

// start launches the fan-out goroutine exactly once.
func (sr *StreamResult) start() {
	go func() {
		defer close(sr.done)
		defer close(sr.textCh)
		defer close(sr.eventsCh)
		defer close(sr.consumeCh)

		for ev := range sr.src {
			// Fan out to all three channels.
			if ev.Type == engine.StepEventTextDelta {
				sr.textCh <- ev.TextDelta
			}
			sr.eventsCh <- ev
			sr.consumeCh <- ev
		}
	}()
}

// TextStream returns a channel yielding text deltas only.
// The channel is closed when the stream completes.
func (sr *StreamResult) TextStream() <-chan string {
	return sr.textCh
}

// Events returns the raw engine StepEvent channel.
// This is an escape hatch for callers such as uistream.Adapter that need
// full event visibility.
// The channel is closed when the stream completes.
func (sr *StreamResult) Events() <-chan engine.StepEvent {
	return sr.eventsCh
}

// DrainUnused starts goroutines to consume channels that won't be read.
// Call this when only Events() is being consumed (e.g. from StreamToWriter)
// to prevent the fan-out goroutine from deadlocking on full buffers.
func (sr *StreamResult) DrainUnused() {
	go func() {
		for range sr.textCh {
		}
	}()
	go func() {
		for range sr.consumeCh {
		}
	}()
}

// Consume blocks until the stream completes and returns the aggregated result.
// It reads from its own internal channel so it does not interfere with
// TextStream or Events consumers.
func (sr *StreamResult) Consume() (*GenerateTextResult, error) {
	result := &GenerateTextResult{}
	var currentStep *StepOutput

	for ev := range sr.consumeCh {
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
