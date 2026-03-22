package ai

import (
	"sync"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// StreamResult wraps a streaming response with convenient accessors.
// It fans out the source engine channel to text-delta and raw-event subscribers.
//
// Callers must consume at least one channel (Events, TextStream, or Consume)
// or call DrainUnused/ConsumeStream to prevent goroutine leaks.
//
// The fan-out goroutine is started lazily on the first call to Events(),
// TextStream(), Consume(), DrainUnused(), or ConsumeStream(). Channels that
// are not requested before the fan-out starts are automatically drained, so
// callers never need to worry about deadlocks from unconsumed channels.
type StreamResult struct {
	src <-chan engine.StepEvent

	textCh    chan string
	eventsCh  chan engine.StepEvent
	consumeCh chan engine.StepEvent

	// done is closed when the fan-out goroutine has finished.
	done chan struct{}

	// Track which channels were requested before fan-out starts.
	mu               sync.Mutex
	startOnce        sync.Once
	drainOnce        sync.Once
	consumeOnce      sync.Once
	textRequested    bool
	eventsRequested  bool
	consumeRequested bool
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
	return sr
}

// ensureStarted launches the fan-out goroutine exactly once.
// It snapshots which channels have been requested and skips unrequested ones.
func (sr *StreamResult) ensureStarted() {
	sr.startOnce.Do(func() {
		sr.mu.Lock()
		wantText := sr.textRequested
		wantEvents := sr.eventsRequested
		wantConsume := sr.consumeRequested
		sr.mu.Unlock()

		go func() {
			defer close(sr.done)
			defer close(sr.textCh)
			defer close(sr.eventsCh)
			defer close(sr.consumeCh)

			for ev := range sr.src {
				if ev.Type == engine.StepEventTextDelta && wantText {
					sr.textCh <- ev.TextDelta
				}
				if wantEvents {
					sr.eventsCh <- ev
				}
				if wantConsume {
					sr.consumeCh <- ev
				}
			}
		}()
	})
}

// TextStream returns a channel yielding text deltas only.
// The channel is closed when the stream completes.
func (sr *StreamResult) TextStream() <-chan string {
	sr.mu.Lock()
	sr.textRequested = true
	sr.mu.Unlock()
	sr.ensureStarted()
	return sr.textCh
}

// Events returns the raw engine StepEvent channel.
// This is an escape hatch for callers such as uistream.Adapter that need
// full event visibility.
// The channel is closed when the stream completes.
func (sr *StreamResult) Events() <-chan engine.StepEvent {
	sr.mu.Lock()
	sr.eventsRequested = true
	sr.mu.Unlock()
	sr.ensureStarted()
	return sr.eventsCh
}

// DrainUnused starts goroutines to consume channels that won't be read.
// Call this when only Events() is being consumed (e.g. from StreamToWriter)
// to prevent the fan-out goroutine from deadlocking on full buffers.
//
// Safe to call multiple times; only the first call spawns drain goroutines.
// Must not be combined with Consume() — both read from consumeCh.
//
// NOTE: With the lazy fan-out design, DrainUnused is no longer strictly
// required — unrequested channels are automatically skipped. It is kept
// for backward compatibility and as an explicit safety net.
func (sr *StreamResult) DrainUnused() {
	sr.drainOnce.Do(func() {
		// Mark textCh and consumeCh as requested so the fan-out sends to them,
		// then drain them in background goroutines.
		// NOTE: Do NOT call ensureStarted() here — the caller will call Events()
		// (or TextStream/Consume) which triggers ensureStarted with all flags set.
		sr.mu.Lock()
		sr.textRequested = true
		sr.consumeRequested = true
		sr.mu.Unlock()

		go func() {
			for range sr.textCh {
			}
		}()
		go func() {
			for range sr.consumeCh {
			}
		}()
	})
}

// ConsumeStream drains all output channels for fire-and-forget usage.
// Call this when you want the stream to run to completion without reading
// any output — e.g. when side effects are handled via callbacks or merge.
//
// Unlike DrainUnused (which preserves Events() for reading), ConsumeStream
// drains everything including the events channel.
//
// Must not be combined with DrainUnused or direct channel reads.
func (sr *StreamResult) ConsumeStream() {
	sr.consumeOnce.Do(func() {
		sr.mu.Lock()
		sr.textRequested = true
		sr.eventsRequested = true
		sr.consumeRequested = true
		sr.mu.Unlock()
		sr.ensureStarted()
		go func() {
			for range sr.textCh {
			}
		}()
		go func() {
			for range sr.eventsCh {
			}
		}()
		go func() {
			for range sr.consumeCh {
			}
		}()
	})
}

// Consume blocks until the stream completes and returns the aggregated result.
// It reads from its own internal channel so it does not interfere with
// TextStream or Events consumers.
func (sr *StreamResult) Consume() (*GenerateTextResult, error) {
	sr.mu.Lock()
	sr.consumeRequested = true
	sr.mu.Unlock()
	sr.ensureStarted()

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
