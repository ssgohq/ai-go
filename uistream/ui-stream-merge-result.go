package uistream

import (
	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// StreamEventer is satisfied by *ai.StreamResult; using an interface avoids
// an import cycle between the uistream and ai packages.
type StreamEventer interface {
	Events() <-chan engine.StepEvent
	// DrainUnused prevents fan-out deadlocks when only Events() is consumed.
	DrainUnused()
}

// mergeConfig holds options for MergeStreamResult.
type mergeConfig struct {
	toolResultHook ToolResultHook
	sourceHook     SourceHook
	onFinish       func(text string)
}

// MergeOption configures MergeStreamResult behavior.
type MergeOption func(*mergeConfig)

// MergeWithToolResultHook sets a hook called after each tool result is emitted.
func MergeWithToolResultHook(hook ToolResultHook) MergeOption {
	return func(c *mergeConfig) {
		c.toolResultHook = hook
	}
}

// MergeWithSourceHook sets a callback invoked when a source-url chunk is emitted.
func MergeWithSourceHook(hook SourceHook) MergeOption {
	return func(c *mergeConfig) {
		c.sourceHook = hook
	}
}

// MergeWithOnFinish sets a callback invoked when the merged stream completes.
func MergeWithOnFinish(fn func(text string)) MergeOption {
	return func(c *mergeConfig) {
		c.onFinish = fn
	}
}

// MergeStreamResult pipes events from sr through this Writer using a temporary
// Adapter. Custom chunks can be written to wr before and after the call.
// Returns the full accumulated assistant text.
//
// Example:
//
//	wr := uistream.NewWriter(sseWriter)
//	wr.WriteStart(msgID)
//	wr.WriteData("plan", planData)           // custom data before stream
//	text := wr.MergeStreamResult(result)     // model stream events
//	wr.WriteData("sources", sourcesData)     // custom data after stream
//	wr.WriteFinish()
//
// Note: MergeStreamResult does NOT emit start or finish chunks; lifecycle
// management (WriteStart / WriteFinish) remains the caller's responsibility.
func (wr *Writer) MergeStreamResult(sr StreamEventer, opts ...MergeOption) string {
	cfg := &mergeConfig{}
	for _, o := range opts {
		o(cfg)
	}

	// Use a bare ChunkProducer (no msgID) so it does not emit a duplicate start
	// chunk — the caller manages start/finish lifecycle.
	producer := newMergeProducer()

	// Drain unused channels to prevent fan-out goroutine deadlock.
	sr.DrainUnused()

	ch := sr.Events()

	// If a tool result hook is set, intercept events before the producer.
	producerCh := ch
	type toolData struct {
		toolName string
		argsJSON string
		output   string
	}
	var toolCache map[string]toolData
	if cfg.toolResultHook != nil {
		toolCache = make(map[string]toolData)
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

	cs := producer.Produce(producerCh)

	for c := range cs.Chunks {
		switch c.Type {
		case ChunkFinish:
			// Do NOT emit finish here; caller manages lifecycle.
		case ChunkStart:
			// Skip the start chunk emitted by the producer; caller owns lifecycle.
		case ChunkError:
			msg, ok := c.Fields["errorText"].(string)
			if !ok {
				msg = "stream error"
			}
			wr.WriteError(msg)
		default:
			wr.WriteChunk(c.Type, c.Fields)

			if c.Type == ChunkSourceURL && cfg.sourceHook != nil {
				sid, _ := c.Fields["sourceId"].(string)
				surl, _ := c.Fields["url"].(string)
				stitle, _ := c.Fields["title"].(string)
				cfg.sourceHook(wr, sid, surl, stitle)
			}

			if c.Type == ChunkToolOutputAvailable && cfg.toolResultHook != nil {
				tcID, ok := c.Fields["toolCallId"].(string)
				if ok {
					if td, found := toolCache[tcID]; found {
						cfg.toolResultHook(wr, ToolResult{
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

	if cfg.onFinish != nil {
		cfg.onFinish(text)
	}

	return text
}

// newMergeProducer creates a ChunkProducer with an empty msgID since the
// merge flow does not emit start/finish lifecycle chunks.
func newMergeProducer() *ChunkProducer {
	return NewChunkProducer("")
}
