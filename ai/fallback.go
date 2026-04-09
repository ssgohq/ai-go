package ai

import (
	"context"
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
func WithFallback(models ...LanguageModel) LanguageModel {
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
func (f *FallbackModel) Stream(ctx context.Context, req LanguageModelRequest) (<-chan StreamEvent, error) {
	var lastErr error
	for i, model := range f.models {
		ch, err := model.Stream(ctx, req)
		if err == nil {
			return ch, nil
		}
		lastErr = err
		if !isFallbackRetryable(err) {
			return nil, fmt.Errorf("fallback model %d (%s): %w", i, model.ModelID(), err)
		}
	}
	return nil, fmt.Errorf("all fallback models failed, last error: %w", lastErr)
}

// isFallbackRetryable returns true for errors that warrant trying the next model.
func isFallbackRetryable(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	// Transient: rate limit, server errors, timeouts
	for _, code := range []string{"429", "500", "502", "503", "529"} {
		if strings.Contains(s, "status "+code) || strings.Contains(s, "unexpected status "+code) {
			return true
		}
	}
	if strings.Contains(s, "i/o timeout") || strings.Contains(s, "context deadline exceeded") {
		return true
	}
	return false
}
