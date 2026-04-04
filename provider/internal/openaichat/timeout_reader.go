package openaichat

import (
	"errors"
	"io"
	"sync"
	"time"
)

// ErrChunkTimeout is returned when no data is received within the configured chunk timeout.
var ErrChunkTimeout = errors.New("SSE chunk read timed out")

// timeoutReader wraps an io.ReadCloser and closes it if no Read() call
// completes within the configured timeout. Each successful Read resets the timer.
type timeoutReader struct {
	inner   io.ReadCloser
	timeout time.Duration
	timer   *time.Timer
	once    sync.Once
	closed  chan struct{}
}

// NewTimeoutReader creates a timeoutReader that wraps r and aborts if no data
// arrives within the given timeout. Each successful Read resets the timer.
func NewTimeoutReader(r io.ReadCloser, timeout time.Duration) *timeoutReader {
	tr := &timeoutReader{
		inner:   r,
		timeout: timeout,
		closed:  make(chan struct{}),
	}
	tr.timer = time.AfterFunc(timeout, func() {
		tr.once.Do(func() {
			close(tr.closed)
			tr.inner.Close()
		})
	})
	return tr
}

func (r *timeoutReader) Read(p []byte) (int, error) {
	n, err := r.inner.Read(p)
	if n > 0 {
		r.timer.Reset(r.timeout)
	}
	if err != nil {
		r.timer.Stop()
		select {
		case <-r.closed:
			return n, ErrChunkTimeout
		default:
		}
	}
	return n, err
}

func (r *timeoutReader) Close() error {
	r.timer.Stop()
	return r.inner.Close()
}
