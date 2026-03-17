package uistream

import (
	"io"

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

// Stream consumes events from ch and writes SSE lines to w.
// It returns the full concatenated assistant text for persistence.
//
// When a ToolResultHook is set, it intercepts engine.StepEventToolResult events
// to capture raw arg/output strings before forwarding to ChunkProducer.
func (a *Adapter) Stream(ch <-chan engine.StepEvent, w io.Writer) string {
	// toolCache stores raw tool result strings keyed by toolCallID so the
	// ToolResultHook can access unmodified Args and Output.
	type toolData struct {
		toolName string
		argsJSON string
		output   string
	}
	var toolCache map[string]toolData
	if a.toolResultHook != nil {
		toolCache = make(map[string]toolData)
	}

	// If a hook is registered, wrap ch to intercept tool result events.
	producerCh := ch
	if a.toolResultHook != nil {
		intercepted := make(chan engine.StepEvent, 64)
		go func() {
			defer close(intercepted)
			for ev := range ch {
				if ev.Type == engine.StepEventToolResult && ev.ToolResult != nil {
					tr := ev.ToolResult
					toolCache[tr.ID] = toolData{
						toolName: tr.Name,
						argsJSON: tr.Args,
						output:   tr.Output,
					}
				}
				intercepted <- ev
			}
		}()
		producerCh = intercepted
	}

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
			wr.WriteFinish()
		case ChunkError:
			msg, ok := c.Fields["errorText"].(string)
			if !ok {
				msg = "stream error"
			}
			wr.WriteError(msg)
		default:
			wr.WriteChunk(c.Type, c.Fields)

			// Fire source hook when a source-url chunk is emitted.
			if c.Type == ChunkSourceURL && a.sourceHook != nil {
				sid, ok1 := c.Fields["sourceId"].(string)
				surl, ok2 := c.Fields["url"].(string)
				stitle, ok3 := c.Fields["title"].(string)
				_ = ok1
				_ = ok2
				_ = ok3
				a.sourceHook(wr, sid, surl, stitle)
			}

			// Fire hook after tool-output-available with raw string data.
			if c.Type == ChunkToolOutputAvailable && a.toolResultHook != nil {
				tcID, ok := c.Fields["toolCallId"].(string)
				if ok {
					if td, found := toolCache[tcID]; found {
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
	}

	text := cs.FullText()

	if a.onFinish != nil {
		a.onFinish(text, lastFinishReason)
	}

	return text
}
