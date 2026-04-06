package ai

import (
	"context"
	"testing"
	"time"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// feedEvents sends events to a channel then closes it.
func feedEvents(events ...engine.StepEvent) <-chan engine.StepEvent {
	ch := make(chan engine.StepEvent, len(events))
	for _, ev := range events {
		ch <- ev
	}
	close(ch)
	return ch
}

// collectEvents drains a channel into a slice with a timeout.
func collectEvents(t *testing.T, ch <-chan engine.StepEvent) []engine.StepEvent {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var result []engine.StepEvent
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				return result
			}
			result = append(result, ev)
		case <-ctx.Done():
			t.Fatal("collectEvents timed out")
			return result
		}
	}
}

func textDelta(s string) engine.StepEvent {
	return engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: s}
}

func reasoningDelta(s string) engine.StepEvent {
	return engine.StepEvent{Type: engine.StepEventReasoningDelta, ReasoningDelta: s}
}

func stepStart() engine.StepEvent {
	return engine.StepEvent{Type: engine.StepEventStepStart}
}

func stepEnd() engine.StepEvent {
	return engine.StepEvent{Type: engine.StepEventStepEnd}
}

func toolCallStart(name string) engine.StepEvent {
	return engine.StepEvent{Type: engine.StepEventToolCallStart, ToolCallName: name}
}

func TestWordChunkingCombinesPartialWords(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(0))
	in := feedEvents(textDelta("Hello "), textDelta("wor"), textDelta("ld "))
	out := collectEvents(t, ss.Transform(context.Background(), in))

	var text string
	for _, ev := range out {
		if ev.Type == engine.StepEventTextDelta {
			text += ev.TextDelta
		}
	}
	if text != "Hello world " {
		t.Errorf("expected 'Hello world ', got %q", text)
	}

	// Should emit "Hello " and "world " as separate chunks
	var deltas []string
	for _, ev := range out {
		if ev.Type == engine.StepEventTextDelta {
			deltas = append(deltas, ev.TextDelta)
		}
	}
	if len(deltas) != 2 {
		t.Errorf("expected 2 word chunks, got %d: %v", len(deltas), deltas)
	}
}

func TestWordChunkingFlushesOnStreamEnd(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(0))
	in := feedEvents(textDelta("fragment"))
	out := collectEvents(t, ss.Transform(context.Background(), in))

	if len(out) != 1 || out[0].TextDelta != "fragment" {
		t.Errorf("expected single flush of 'fragment', got %v", out)
	}
}

func TestLineChunkingSplitsOnNewlines(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(0), WithLineChunking())
	in := feedEvents(textDelta("line1\nline2\n"))
	out := collectEvents(t, ss.Transform(context.Background(), in))

	var deltas []string
	for _, ev := range out {
		if ev.Type == engine.StepEventTextDelta {
			deltas = append(deltas, ev.TextDelta)
		}
	}
	if len(deltas) != 2 {
		t.Errorf("expected 2 line chunks, got %d: %v", len(deltas), deltas)
	}
	if deltas[0] != "line1\n" || deltas[1] != "line2\n" {
		t.Errorf("unexpected chunks: %v", deltas)
	}
}

func TestRegexChunkingCustomPattern(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(0), WithRegexChunking(`[a-z]+`))
	in := feedEvents(textDelta("abc123def456"))
	out := collectEvents(t, ss.Transform(context.Background(), in))

	var deltas []string
	for _, ev := range out {
		if ev.Type == engine.StepEventTextDelta {
			deltas = append(deltas, ev.TextDelta)
		}
	}
	// "abc" matches, then "123def" -> "123def" matches "def" at index 3
	// Remaining "456" is flushed at end
	if len(deltas) < 2 {
		t.Errorf("expected at least 2 chunks, got %d: %v", len(deltas), deltas)
	}
	var total string
	for _, d := range deltas {
		total += d
	}
	if total != "abc123def456" {
		t.Errorf("expected all content preserved, got %q", total)
	}
}

func TestNonTextEventsPassThrough(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(0))
	in := feedEvents(
		stepStart(),
		textDelta("Hello "),
		toolCallStart("calc"),
		textDelta("world"),
		stepEnd(),
	)
	out := collectEvents(t, ss.Transform(context.Background(), in))

	// StepStart should pass through first
	if out[0].Type != engine.StepEventStepStart {
		t.Errorf("expected StepStart first, got %v", out[0].Type)
	}

	// ToolCallStart should flush "Hello " then pass through
	var sawToolCall bool
	for _, ev := range out {
		if ev.Type == engine.StepEventToolCallStart {
			sawToolCall = true
		}
	}
	if !sawToolCall {
		t.Error("ToolCallStart not passed through")
	}

	// All text should be preserved
	var text string
	for _, ev := range out {
		if ev.Type == engine.StepEventTextDelta {
			text += ev.TextDelta
		}
	}
	if text != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", text)
	}
}

func TestReasoningDeltaBufferedSeparately(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(0))
	in := feedEvents(
		textDelta("text "),
		reasoningDelta("reason "),
		textDelta("more "),
	)
	out := collectEvents(t, ss.Transform(context.Background(), in))

	var order []string
	for _, ev := range out {
		switch ev.Type {
		case engine.StepEventTextDelta:
			order = append(order, "T:"+ev.TextDelta)
		case engine.StepEventReasoningDelta:
			order = append(order, "R:"+ev.ReasoningDelta)
		}
	}

	// text buffer should flush before reasoning starts, and reasoning before text resumes
	if len(order) != 3 {
		t.Errorf("expected 3 events, got %d: %v", len(order), order)
	}
	if order[0] != "T:text " {
		t.Errorf("first should be text flush, got %s", order[0])
	}
	if order[1] != "R:reason " {
		t.Errorf("second should be reasoning, got %s", order[1])
	}
	if order[2] != "T:more " {
		t.Errorf("third should be text, got %s", order[2])
	}
}

func TestContextCancellationStopsProcessing(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ss := NewSmoothStream(WithDelayMs(100)) // slow delay to test cancellation

	// Feed many events
	in := make(chan engine.StepEvent, 100)
	for i := 0; i < 50; i++ {
		in <- textDelta("word ")
	}
	close(in)

	out := ss.Transform(ctx, in)

	// Read a couple events then cancel
	<-out
	cancel()

	// Output channel should eventually close without deadlock
	timeout := time.After(2 * time.Second)
	for {
		select {
		case _, ok := <-out:
			if !ok {
				return // success
			}
		case <-timeout:
			t.Fatal("output channel not closed after context cancellation")
		}
	}
}

func TestCustomDetectorValidationEmptyMatch(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(0), WithChunkDetector(func(buffer string) (string, string, error) {
		return "", "", nil // always returns empty — should cause no emission
	}))
	in := feedEvents(textDelta("test"))
	out := collectEvents(t, ss.Transform(context.Background(), in))

	// Detector returns empty = no chunk detected; buffer flushed at end
	if len(out) != 1 || out[0].TextDelta != "test" {
		t.Errorf("expected flush of 'test', got %v", out)
	}
}

func TestCustomDetectorValidationNonPrefix(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(0), WithChunkDetector(func(buffer string) (string, string, error) {
		return "xyz", "", nil // not a prefix of buffer
	}))
	in := feedEvents(textDelta("abc"))
	out := collectEvents(t, ss.Transform(context.Background(), in))

	// Non-prefix error causes flush of entire buffer
	var text string
	for _, ev := range out {
		if ev.Type == engine.StepEventTextDelta {
			text += ev.TextDelta
		}
	}
	if text != "abc" {
		t.Errorf("expected 'abc' flushed, got %q", text)
	}
}

func TestDelayAppliedBetweenChunks(t *testing.T) {
	ss := NewSmoothStream(WithDelayMs(50))
	in := feedEvents(textDelta("a b c ")) // 3 words
	start := time.Now()
	out := collectEvents(t, ss.Transform(context.Background(), in))
	elapsed := time.Since(start)

	var count int
	for _, ev := range out {
		if ev.Type == engine.StepEventTextDelta {
			count++
		}
	}
	if count != 3 {
		t.Errorf("expected 3 chunks, got %d", count)
	}

	// 3 chunks = 3 delays (after each chunk). Allow generous threshold.
	if elapsed < 100*time.Millisecond {
		t.Errorf("expected >= 100ms elapsed (3 chunks * 50ms delay), got %v", elapsed)
	}
}

func TestDefaultsAreWordChunking10msDelay(t *testing.T) {
	ss := NewSmoothStream()
	if ss.delayMs != 10 {
		t.Errorf("expected default delay 10ms, got %d", ss.delayMs)
	}
	// Verify word chunking works via a quick test
	in := feedEvents(textDelta("hello "))
	out := collectEvents(t, ss.Transform(context.Background(), in))
	if len(out) != 1 || out[0].TextDelta != "hello " {
		t.Errorf("unexpected output: %v", out)
	}
}
