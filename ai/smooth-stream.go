package ai

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"time"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// ChunkDetector extracts the next chunk from a buffer.
// Returns (chunk, remaining, nil) when a chunk is detected.
// Returns ("", "", nil) when no chunk is detected yet (buffer needs more data).
type ChunkDetector func(buffer string) (chunk string, remaining string, err error)

// SmoothStream buffers and re-chunks text-delta and reasoning-delta events
// for a smoother streaming UX. Non-text events pass through immediately.
type SmoothStream struct {
	delayMs  int
	detector ChunkDetector
}

// SmoothStreamOption configures a SmoothStream instance.
type SmoothStreamOption func(*SmoothStream)

// NewSmoothStream creates a SmoothStream with the given options.
// Defaults: 10ms delay, word-level chunking.
func NewSmoothStream(opts ...SmoothStreamOption) *SmoothStream {
	ss := &SmoothStream{
		delayMs:  10,
		detector: wordChunkDetector,
	}
	for _, o := range opts {
		o(ss)
	}
	return ss
}

// WithDelayMs sets the inter-chunk delay in milliseconds. 0 means no delay.
func WithDelayMs(ms int) SmoothStreamOption {
	return func(ss *SmoothStream) { ss.delayMs = ms }
}

// WithWordChunking uses word-level chunking (non-whitespace + trailing whitespace).
func WithWordChunking() SmoothStreamOption {
	return func(ss *SmoothStream) { ss.detector = wordChunkDetector }
}

// WithLineChunking uses line-level chunking (split on newlines).
func WithLineChunking() SmoothStreamOption {
	return func(ss *SmoothStream) { ss.detector = lineChunkDetector }
}

// WithRegexChunking uses a custom regex pattern for chunking.
// Panics if the pattern is invalid.
func WithRegexChunking(pattern string) SmoothStreamOption {
	re := regexp.MustCompile(pattern)
	return func(ss *SmoothStream) {
		ss.detector = makeRegexDetector(re)
	}
}

// WithChunkDetector sets a custom chunk detector function.
func WithChunkDetector(fn ChunkDetector) SmoothStreamOption {
	return func(ss *SmoothStream) {
		ss.detector = func(buffer string) (string, string, error) {
			chunk, remaining, err := fn(buffer)
			if err != nil {
				return "", "", err
			}
			if chunk == "" {
				return "", "", nil
			}
			if !strings.HasPrefix(buffer, chunk) {
				return "", "", errors.New("smooth-stream: chunk detector must return a prefix of the buffer")
			}
			return chunk, remaining, nil
		}
	}
}

// Transform pipes an input event channel through smooth-stream buffering
// and returns a new channel of re-chunked events.
func (ss *SmoothStream) Transform(ctx context.Context, in_ <-chan engine.StepEvent) <-chan engine.StepEvent {
	out := make(chan engine.StepEvent, 64)
	go func() {
		defer close(out)

		var buffer string
		var bufType engine.StepEventType
		var bufActive bool

		send := func(ev engine.StepEvent) bool {
			select {
			case out <- ev:
				return true
			case <-ctx.Done():
				return false
			}
		}

		flush := func() bool {
			if buffer == "" || !bufActive {
				return true
			}
			var ev engine.StepEvent
			ev.Type = bufType
			switch bufType {
			case engine.StepEventTextDelta:
				ev.TextDelta = buffer
			case engine.StepEventReasoningDelta:
				ev.ReasoningDelta = buffer
			}
			buffer = ""
			bufActive = false
			return send(ev)
		}

		emitChunks := func() bool {
			for {
				if buffer == "" {
					return true
				}
				chunk, remaining, err := ss.detector(buffer)
				if err != nil {
					// On detector error, flush entire buffer
					return flush()
				}
				if chunk == "" {
					return true
				}

				var ev engine.StepEvent
				ev.Type = bufType
				switch bufType {
				case engine.StepEventTextDelta:
					ev.TextDelta = chunk
				case engine.StepEventReasoningDelta:
					ev.ReasoningDelta = chunk
				}
				buffer = remaining
				if !send(ev) {
					return false
				}

				if ss.delayMs > 0 {
					select {
					case <-time.After(time.Duration(ss.delayMs) * time.Millisecond):
					case <-ctx.Done():
						return false
					}
				}
			}
		}

		for ev := range in_ {
			switch ev.Type {
			case engine.StepEventTextDelta:
				// Flush if type changed
				if bufType != engine.StepEventTextDelta && buffer != "" {
					if !flush() {
						return
					}
				}
				buffer += ev.TextDelta
				bufType = engine.StepEventTextDelta
				bufActive = true
				if !emitChunks() {
					return
				}

			case engine.StepEventReasoningDelta:
				if bufType != engine.StepEventReasoningDelta && buffer != "" {
					if !flush() {
						return
					}
				}
				buffer += ev.ReasoningDelta
				bufType = engine.StepEventReasoningDelta
				bufActive = true
				if !emitChunks() {
					return
				}

			default:
				// Non-text event: flush buffer, pass through
				if !flush() {
					return
				}
				if !send(ev) {
					return
				}
			}
		}

		// Stream ended: flush remaining buffer
		flush()
	}()
	return out
}

// --- Built-in chunk detectors ---

var wordRegex = regexp.MustCompile(`\S+\s+`)
var lineRegex = regexp.MustCompile(`\n+`)

func wordChunkDetector(buffer string) (string, string, error) {
	return regexDetect(wordRegex, buffer)
}

func lineChunkDetector(buffer string) (string, string, error) {
	return regexDetect(lineRegex, buffer)
}

func makeRegexDetector(re *regexp.Regexp) ChunkDetector {
	return func(buffer string) (string, string, error) {
		return regexDetect(re, buffer)
	}
}

// regexDetect finds the first match in buffer and returns everything up to
// and including the match as the chunk. This intentionally includes any
// unmatched prefix before the match (e.g. "123def" with /[a-z]+/ returns
// "123def"), matching the Node.js reference: buffer.slice(0, match.index) + match[0].
func regexDetect(re *regexp.Regexp, buffer string) (string, string, error) {
	loc := re.FindStringIndex(buffer)
	if loc == nil {
		return "", "", nil
	}
	end := loc[1]
	chunk := buffer[:end]
	remaining := buffer[end:]
	return chunk, remaining, nil
}
