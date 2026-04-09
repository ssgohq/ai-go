package ai

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// RetryConfig controls retry behavior for transient LLM provider errors.
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts. Default: 2.
	// Use a pointer to distinguish "unset" from explicit 0.
	MaxRetries *int
	// InitialDelay is the base delay before the first retry. Default: 1s.
	InitialDelay time.Duration
	// MaxDelay caps the exponential backoff. Default: 60s.
	MaxDelay time.Duration
	// BackoffFactor multiplies the delay each retry. Default: 2.0.
	BackoffFactor float64
	// Jitter adds randomness to prevent thundering herd. Default: true.
	Jitter *bool
	// OnRetry is called before each retry attempt with the attempt number (1-based)
	// and the error that triggered it. Optional.
	OnRetry func(attempt int, err error)
}

// retryDefaults holds the resolved retry configuration with all defaults applied.
type retryDefaults struct {
	maxRetries    int
	initialDelay  time.Duration
	maxDelay      time.Duration
	backoffFactor float64
	jitter        bool
	onRetry       func(attempt int, err error)
}

func (c RetryConfig) resolve() retryDefaults {
	d := retryDefaults{
		maxRetries:    2,
		initialDelay:  time.Second,
		maxDelay:      60 * time.Second,
		backoffFactor: 2.0,
		jitter:        true,
		onRetry:       c.OnRetry,
	}
	if c.MaxRetries != nil {
		d.maxRetries = *c.MaxRetries
	}
	if c.InitialDelay > 0 {
		d.initialDelay = c.InitialDelay
	}
	if c.MaxDelay > 0 {
		d.maxDelay = c.MaxDelay
	}
	if c.BackoffFactor > 0 {
		d.backoffFactor = c.BackoffFactor
	}
	if c.Jitter != nil {
		d.jitter = *c.Jitter
	}
	return d
}

// intPtr returns a pointer to n.
func intPtr(n int) *int { return &n }

// boolPtr returns a pointer to b.
func boolPtr(b bool) *bool { return &b }

// WithMaxRetries returns an Option that enables retry with the given max attempts.
// Uses default backoff settings (1s initial, 2x factor, 60s max, jitter enabled).
// Pass 0 to explicitly disable retries.
func WithMaxRetries(n int) Option {
	return WithRetry(RetryConfig{MaxRetries: intPtr(n), Jitter: boolPtr(true)})
}

// WithRetry returns an Option that stores retry config for deferred model wrapping.
func WithRetry(config RetryConfig) Option {
	return func(r *GenerateTextRequest) {
		mw := func(model LanguageModel) LanguageModel {
			return newRetryModel(model, config)
		}
		r.Middlewares = append(r.Middlewares, mw)
	}
}

// retryModel wraps a LanguageModel with retry logic.
type retryModel struct {
	inner  LanguageModel
	config retryDefaults
}

func newRetryModel(inner LanguageModel, config RetryConfig) *retryModel {
	return &retryModel{inner: inner, config: config.resolve()}
}

func (m *retryModel) ModelID() string { return m.inner.ModelID() }

func (m *retryModel) Stream(
	ctx context.Context,
	req LanguageModelRequest,
) (<-chan StreamEvent, error) {
	var lastErr error
	for attempt := 0; attempt <= m.config.maxRetries; attempt++ {
		if attempt > 0 {
			if m.config.onRetry != nil {
				m.config.onRetry(attempt, lastErr)
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
	delay := float64(m.config.initialDelay) * math.Pow(
		m.config.backoffFactor, float64(attempt-1),
	)
	if delay > float64(m.config.maxDelay) {
		delay = float64(m.config.maxDelay)
	}
	if m.config.jitter {
		delay = delay * (0.5 + cryptoFloat64()*0.5)
	}
	return time.Duration(delay)
}

// isRetryable returns true for transient HTTP errors worth retrying.
func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	for _, code := range []string{"429", "500", "502", "503", "529"} {
		if strings.Contains(s, "status "+code) ||
			strings.Contains(s, "unexpected status "+code) {
			return true
		}
	}
	if strings.Contains(s, "connection refused") ||
		strings.Contains(s, "connection reset") ||
		strings.Contains(s, "i/o timeout") ||
		strings.Contains(s, "EOF") {
		return true
	}
	return false
}

// parseRetryAfter extracts a Retry-After duration from an error message.
// Works when the provider includes the response body in the error string.
func parseRetryAfter(err error) time.Duration {
	if err == nil {
		return 0
	}
	s := err.Error()
	idx := strings.Index(strings.ToLower(s), "retry-after")
	if idx < 0 {
		return 0
	}
	rest := s[idx:]
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

// cryptoFloat64 returns a cryptographically random float64 in [0, 1).
func cryptoFloat64() float64 {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return float64(binary.LittleEndian.Uint64(b[:])>>11) / (1 << 53)
}
