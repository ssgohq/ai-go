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
type ChunkDetector func(buffer string) (chunk, remaining string, err error)

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
		s := &smoothState{ss: ss, ctx: ctx, out: out}
		s.run(in_)
	}()
	return out
}

// smoothState holds the mutable state for a single Transform run.
type smoothState struct {
	ss  *SmoothStream
	ctx context.Context
	out chan<- engine.StepEvent

	buffer    string
	bufType   engine.StepEventType
	bufActive bool
}

func (s *smoothState) send(ev engine.StepEvent) bool {
	select {
	case s.out <- ev:
		return true
	case <-s.ctx.Done():
		return false
	}
}

func (s *smoothState) flush() bool {
	if s.buffer == "" || !s.bufActive {
		return true
	}
	ev := s.makeBufferEvent(s.buffer)
	s.buffer = ""
	s.bufActive = false
	return s.send(ev)
}

func (s *smoothState) makeBufferEvent(text string) engine.StepEvent {
	var ev engine.StepEvent
	ev.Type = s.bufType
	if s.bufType == engine.StepEventTextDelta {
		ev.TextDelta = text
	} else {
		ev.ReasoningDelta = text
	}
	return ev
}

func (s *smoothState) emitChunks() bool {
	for s.buffer != "" {
		chunk, remaining, err := s.ss.detector(s.buffer)
		if err != nil {
			return s.flush()
		}
		if chunk == "" {
			return true
		}
		ev := s.makeBufferEvent(chunk)
		s.buffer = remaining
		if !s.send(ev) {
			return false
		}
		if !s.sleepDelay() {
			return false
		}
	}
	return true
}

func (s *smoothState) sleepDelay() bool {
	if s.ss.delayMs <= 0 {
		return true
	}
	select {
	case <-time.After(time.Duration(s.ss.delayMs) * time.Millisecond):
		return true
	case <-s.ctx.Done():
		return false
	}
}

func (s *smoothState) appendDelta(ev engine.StepEvent) bool {
	evType := ev.Type
	delta := ev.TextDelta
	if evType == engine.StepEventReasoningDelta {
		delta = ev.ReasoningDelta
	}
	// Flush if type changed
	if s.bufType != evType && s.buffer != "" {
		if !s.flush() {
			return false
		}
	}
	s.buffer += delta
	s.bufType = evType
	s.bufActive = true
	return s.emitChunks()
}

func (s *smoothState) run(in_ <-chan engine.StepEvent) {
	for ev := range in_ {
		switch ev.Type {
		case engine.StepEventTextDelta, engine.StepEventReasoningDelta:
			if !s.appendDelta(ev) {
				return
			}
		default:
			if !s.flush() || !s.send(ev) {
				return
			}
		}
	}
	s.flush()
}

// --- Built-in chunk detectors ---

var (
	wordRegex = regexp.MustCompile(`\S+\s+`)
	lineRegex = regexp.MustCompile(`\n+`)
)

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
