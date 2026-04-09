package ai

import (
	"context"
	"math"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// RetryConfig controls retry behavior for transient LLM provider errors.
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts. Default: 2.
	MaxRetries int
	// InitialDelay is the base delay before the first retry. Default: 1s.
	InitialDelay time.Duration
	// MaxDelay caps the exponential backoff. Default: 60s.
	MaxDelay time.Duration
	// BackoffFactor multiplies the delay each retry. Default: 2.0.
	BackoffFactor float64
	// Jitter adds randomness to prevent thundering herd. Default: true.
	Jitter bool
	// OnRetry is called before each retry attempt with the attempt number (1-based)
	// and the error that triggered it. Optional.
	OnRetry func(attempt int, err error)
}

func (c RetryConfig) withDefaults() RetryConfig {
	if c.MaxRetries <= 0 {
		c.MaxRetries = 2
	}
	if c.InitialDelay <= 0 {
		c.InitialDelay = time.Second
	}
	if c.MaxDelay <= 0 {
		c.MaxDelay = 60 * time.Second
	}
	if c.BackoffFactor <= 0 {
		c.BackoffFactor = 2.0
	}
	return c
}

// WithMaxRetries returns an Option that enables retry with the given max attempts.
// Uses default backoff settings (1s initial, 2x factor, 60s max, jitter enabled).
func WithMaxRetries(n int) Option {
	return WithRetry(RetryConfig{MaxRetries: n, Jitter: true})
}

// WithRetry returns an Option that wraps the model with retry middleware.
func WithRetry(config RetryConfig) Option {
	return func(r *GenerateTextRequest) {
		if r.Model != nil {
			r.Model = newRetryModel(r.Model, config)
		}
	}
}

// retryModel wraps a LanguageModel with retry logic.
type retryModel struct {
	inner  LanguageModel
	config RetryConfig
}

func newRetryModel(inner LanguageModel, config RetryConfig) *retryModel {
	return &retryModel{inner: inner, config: config.withDefaults()}
}

func (m *retryModel) ModelID() string { return m.inner.ModelID() }

func (m *retryModel) Stream(ctx context.Context, req LanguageModelRequest) (<-chan StreamEvent, error) {
	var lastErr error
	for attempt := 0; attempt <= m.config.MaxRetries; attempt++ {
		if attempt > 0 {
			if m.config.OnRetry != nil {
				m.config.OnRetry(attempt, lastErr)
			}
			delay := m.calculateDelay(attempt)
			if retryAfter := parseRetryAfter(lastErr); retryAfter > 0 {
				delay = retryAfter
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		ch, err := m.inner.Stream(ctx, req)
		if err == nil {
			return ch, nil
		}
		if !isRetryable(err) {
			return nil, err
		}
		lastErr = err
	}
	return nil, lastErr
}

func (m *retryModel) calculateDelay(attempt int) time.Duration {
	delay := float64(m.config.InitialDelay) * math.Pow(m.config.BackoffFactor, float64(attempt-1))
	if delay > float64(m.config.MaxDelay) {
		delay = float64(m.config.MaxDelay)
	}
	if m.config.Jitter {
		delay = delay * (0.5 + rand.Float64()*0.5)
	}
	return time.Duration(delay)
}

// isRetryable returns true for transient HTTP errors worth retrying.
func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	// Check for retryable HTTP status codes in error messages
	for _, code := range []string{"429", "500", "502", "503", "529"} {
		if strings.Contains(s, "status "+code) || strings.Contains(s, "unexpected status "+code) {
			return true
		}
	}
	// Network-level errors
	if strings.Contains(s, "connection refused") ||
		strings.Contains(s, "connection reset") ||
		strings.Contains(s, "i/o timeout") ||
		strings.Contains(s, "EOF") {
		return true
	}
	return false
}

// parseRetryAfter extracts a Retry-After duration from an error message.
func parseRetryAfter(err error) time.Duration {
	if err == nil {
		return 0
	}
	s := err.Error()
	// Look for "Retry-After: N" in the error body
	idx := strings.Index(strings.ToLower(s), "retry-after")
	if idx < 0 {
		return 0
	}
	rest := s[idx:]
	// Try to find a number after the header name
	for i := 0; i < len(rest); i++ {
		if rest[i] >= '0' && rest[i] <= '9' {
			end := i
			for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
				end++
			}
			if secs, parseErr := strconv.Atoi(rest[i:end]); parseErr == nil {
				return time.Duration(secs) * time.Second
			}
			break
		}
	}
	return 0
}

// RetryableStatusCode checks if an HTTP status code is retryable.
// Exported for use by provider implementations.
func RetryableStatusCode(code int) bool {
	switch code {
	case http.StatusTooManyRequests, // 429
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		529:                            // Anthropic overload
		return true
	}
	return false
}
