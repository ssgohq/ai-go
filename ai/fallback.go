package ai

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// FallbackModel tries models in order, falling back on transient errors.
type FallbackModel struct {
	models []LanguageModel
}

// WithFallback creates a LanguageModel that tries each model in order.
// On transient errors (429, 500+, timeout), the next model is tried.
// Non-transient errors (400, 401, 403) are returned immediately.
// Panics if no models are provided.
func WithFallback(models ...LanguageModel) LanguageModel {
	if len(models) == 0 {
		panic("ai.WithFallback: at least one model is required")
	}
	if len(models) == 1 {
		return models[0]
	}
	return &FallbackModel{models: models}
}

// ModelID returns a composite ID of all fallback models.
func (f *FallbackModel) ModelID() string {
	ids := make([]string, len(f.models))
	for i, m := range f.models {
		ids[i] = m.ModelID()
	}
	return "fallback(" + strings.Join(ids, ",") + ")"
}

// Stream tries each model in order until one succeeds or all fail.
// It handles both synchronous errors (from model.Stream) and asynchronous
// stream errors (StreamEventError on the channel) for the first event only.
func (f *FallbackModel) Stream(ctx context.Context, req LanguageModelRequest) (<-chan StreamEvent, error) {
	var lastErr error
	for i, model := range f.models {
		ch, err := model.Stream(ctx, req)
		if err != nil {
			lastErr = err
			if !isFallbackRetryable(err) {
				return nil, fmt.Errorf(
					"fallback model %d (%s): %w",
					i, model.ModelID(), err,
				)
			}
			continue
		}

		// Peek at the first event to detect immediate stream errors.
		// If the first event is an error and retryable, try the next model.
		firstEvent, ok := <-ch
		if !ok {
			// Channel closed immediately — empty stream, return it.
			out := make(chan StreamEvent)
			close(out)
			return out, nil
		}
		if firstEvent.Type == StreamEventError && firstEvent.Error != nil {
			if isFallbackRetryable(firstEvent.Error) {
				lastErr = firstEvent.Error
				// Drain remaining events to prevent goroutine leak
				go func() {
					for range ch {
					}
				}()
				continue
			}
		}

		// Re-emit the first event followed by the rest of the stream.
		out := make(chan StreamEvent, 64)
		go func() {
			defer close(out)
			out <- firstEvent
			for ev := range ch {
				out <- ev
			}
		}()
		return out, nil
	}
	return nil, fmt.Errorf(
		"all fallback models failed, last error: %w",
		lastErr,
	)
}

// isFallbackRetryable returns true for errors that warrant trying the next model.
func isFallbackRetryable(err error) bool {
	if err == nil {
		return false
	}
	// Don't retry on context cancellation
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	s := err.Error()
	// Transient: rate limit, server errors, timeouts
	for _, code := range []string{"429", "500", "502", "503", "529"} {
		if strings.Contains(s, "status "+code) ||
			strings.Contains(s, "unexpected status "+code) {
			return true
		}
	}
	if strings.Contains(s, "i/o timeout") ||
		strings.Contains(s, "connection refused") ||
		strings.Contains(s, "connection reset") ||
		strings.Contains(s, "EOF") {
		return true
	}
	return false
}
