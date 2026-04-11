package uistream

import (
	"io"
	"sync"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// ToolResult is a public-facing tool result notification emitted during streaming.
// Callers that need to react to tool results (e.g. emit document references) receive
// these via the ToolResultHook set on the Adapter.
type ToolResult struct {
	ToolCallID string
	ToolName   string
	ArgsJSON   string
	Output     string
}

// ToolResultHook is called after a tool result is emitted to the stream.
// wr is bound to the same io.Writer as the adapter so callers can emit additional chunks.
type ToolResultHook func(wr *Writer, result ToolResult)

// SourceHook is called when a source-url chunk is emitted during streaming.
// Callers can use this to collect grounding sources for persistence.
type SourceHook func(wr *Writer, sourceID, url, title string)

// Adapter translates a channel of engine.StepEvents into UI message stream chunks.
// It is transport-agnostic: callers can write to an http.ResponseWriter, a buffer, etc.
//
// For custom data-* chunks or source chunks between or after stream events,
// obtain the Writer via adapter.Writer(w) and call WriteData / WriteSource directly.
type Adapter struct {
	msgID              string
	toolResultHook     ToolResultHook
	sourceHook         SourceHook
	onFinish           func(text, finishReason string)
	persistenceBuilder *PersistedMessageBuilder
}

// NewAdapter creates an Adapter with a fixed message ID.
func NewAdapter(msgID string) *Adapter {
	return &Adapter{msgID: msgID}
}

// WithToolResultHook sets a hook called after each tool result is emitted.
// Use this to emit custom side-channel data (e.g. document references) without
// wrapping the internal event channel.
func (a *Adapter) WithToolResultHook(hook ToolResultHook) *Adapter {
	a.toolResultHook = hook
	return a
}

// WithSourceHook sets a hook called when a source-url chunk is emitted.
// Use this to collect grounding sources for persistence or post-processing.
func (a *Adapter) WithSourceHook(hook SourceHook) *Adapter {
	a.sourceHook = hook
	return a
}

// WithOnFinish sets a callback invoked after the stream completes.
// text is the full accumulated assistant text; finishReason is "stop" or the
// finish reason captured from the last finish chunk.
func (a *Adapter) WithOnFinish(fn func(text, finishReason string)) *Adapter {
	a.onFinish = fn
	return a
}

// WithPersistenceBuilder sets a PersistedMessageBuilder that observes every chunk
// during Stream for typed-parts persistence.
func (a *Adapter) WithPersistenceBuilder(b *PersistedMessageBuilder) *Adapter {
	a.persistenceBuilder = b
	return a
}

// Writer returns a direct-write Writer bound to w for emitting custom chunks
// alongside a stream (e.g. WriteData, WriteSource).
func (a *Adapter) Writer(w io.Writer) *Writer {
	return NewWriter(w)
}

// interceptState holds shared mutable state between the intercept goroutine and
// the main Stream loop, protected by a mutex.
type interceptState struct {
	mu         sync.Mutex
	totalUsage UsageInfo
	toolCache  map[string]toolData
}

// interceptEvents wraps an event channel to track usage and cache tool results.
// The returned channel forwards all events unchanged.
func (a *Adapter) interceptEvents(
	ch <-chan engine.StepEvent,
	state *interceptState,
) <-chan engine.StepEvent {
	intercepted := make(chan engine.StepEvent, 64)
	go func() {
		defer close(intercepted)
		for ev := range ch {
			if ev.Type == engine.StepEventUsage && ev.Usage != nil {
				state.mu.Lock()
				state.totalUsage.PromptTokens += ev.Usage.PromptTokens
				state.totalUsage.CompletionTokens += ev.Usage.CompletionTokens
				state.totalUsage.TotalTokens += ev.Usage.TotalTokens
				state.totalUsage.ReasoningTokens += ev.Usage.ReasoningTokens
				state.mu.Unlock()
			}
			if state.toolCache != nil && ev.Type == engine.StepEventToolResult && ev.ToolResult != nil {
				tr := ev.ToolResult
				state.mu.Lock()
				state.toolCache[tr.ID] = toolData{
					toolName: tr.Name,
					argsJSON: tr.Args,
					output:   tr.Output,
				}
				state.mu.Unlock()
			}
			intercepted <- ev
		}
	}()
	return intercepted
}

// writeChunkWithHooks writes a non-finish, non-error chunk and fires registered hooks.
func (a *Adapter) writeChunkWithHooks(wr *Writer, c Chunk, state *interceptState) {
	wr.WriteChunk(c.Type, c.Fields)

	if c.Type == ChunkSourceURL && a.sourceHook != nil {
		sid, ok1 := c.Fields["sourceId"].(string)
		surl, ok2 := c.Fields["url"].(string)
		stitle, ok3 := c.Fields["title"].(string)
		_, _, _ = ok1, ok2, ok3
		a.sourceHook(wr, sid, surl, stitle)
	}

	if c.Type == ChunkToolOutputAvailable && a.toolResultHook != nil {
		if tcID, ok := c.Fields["toolCallId"].(string); ok {
			state.mu.Lock()
			td, found := state.toolCache[tcID]
			state.mu.Unlock()
			if found {
				a.toolResultHook(wr, ToolResult{
					ToolCallID: tcID,
					ToolName:   td.toolName,
					ArgsJSON:   td.argsJSON,
					Output:     td.output,
				})
			}
		}
	}
}

// toolData stores raw tool result strings keyed by toolCallID.
type toolData struct {
	toolName string
	argsJSON string
	output   string
}

// usageMetadata returns messageMetadata containing usage, or nil if no tokens were tracked.
func usageMetadata(u UsageInfo) map[string]any {
	if u.TotalTokens == 0 {
		return nil
	}
	return map[string]any{
		"usage": map[string]any{
			"promptTokens":     u.PromptTokens,
			"completionTokens": u.CompletionTokens,
			"totalTokens":      u.TotalTokens,
			"reasoningTokens":  u.ReasoningTokens,
		},
	}
}

// Stream consumes events from ch and writes SSE lines to w.
// It returns the full concatenated assistant text for persistence.
func (a *Adapter) Stream(ch <-chan engine.StepEvent, w io.Writer) string {
	state := &interceptState{}
	if a.toolResultHook != nil {
		state.toolCache = make(map[string]toolData)
	}

	producerCh := a.interceptEvents(ch, state)

	producer := NewChunkProducer(a.msgID)
	cs := producer.Produce(producerCh)
	wr := NewWriter(w)

	var lastFinishReason string
	for c := range cs.Chunks {
		if a.persistenceBuilder != nil {
			a.persistenceBuilder.ObserveChunk(c)
		}
		switch c.Type {
		case ChunkFinish:
			if reason, ok := c.Fields["finishReason"].(string); ok {
				lastFinishReason = reason
			}
			state.mu.Lock()
			usage := state.totalUsage
			state.mu.Unlock()
			wr.WriteFinishWithReason(lastFinishReason, usageMetadata(usage))
		case ChunkError:
			msg, ok := c.Fields["errorText"].(string)
			if !ok {
				msg = "stream error"
			}
			wr.WriteError(msg)
		default:
			a.writeChunkWithHooks(wr, c, state)
		}
	}

	text := cs.FullText()

	if a.onFinish != nil {
		a.onFinish(text, lastFinishReason)
	}

	return text
}
