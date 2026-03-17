package uistream

import (
	"io"

	"github.com/open-ai-sdk/ai-go/ai"
)

// UIStreamOption configures StreamToWriter behavior.
type UIStreamOption func(*uiStreamBridgeConfig)

// uiStreamBridgeConfig holds options for StreamToWriter.
type uiStreamBridgeConfig struct {
	toolResultHook ToolResultHook
	sourceHook     SourceHook
	onFinish       func(text string)
}

// WithUIToolResultHook sets a callback invoked after each tool result is emitted.
func WithUIToolResultHook(hook ToolResultHook) UIStreamOption {
	return func(c *uiStreamBridgeConfig) {
		c.toolResultHook = hook
	}
}

// WithUISourceHook sets a callback invoked when a source-url chunk is emitted.
// Use this to collect grounding sources for persistence.
func WithUISourceHook(hook SourceHook) UIStreamOption {
	return func(c *uiStreamBridgeConfig) {
		c.sourceHook = hook
	}
}

// WithUIOnFinish sets a callback invoked when the stream completes.
// text is the full accumulated assistant response.
func WithUIOnFinish(fn func(text string)) UIStreamOption {
	return func(c *uiStreamBridgeConfig) {
		c.onFinish = fn
	}
}

// StreamToWriter writes SSE UI message stream chunks to w, consuming all events
// from sr. It returns the full assistant text for persistence.
//
// msgID is the assistant message identifier emitted in the start chunk.
// Callers may pass UIStreamOption values to attach hooks.
func StreamToWriter(sr *ai.StreamResult, w io.Writer, msgID string, opts ...UIStreamOption) string {
	cfg := &uiStreamBridgeConfig{}
	for _, o := range opts {
		o(cfg)
	}

	// Drain textCh and consumeCh so the fan-out goroutine doesn't deadlock.
	// StreamToWriter only consumes Events().
	sr.DrainUnused()

	adapter := NewAdapter(msgID)

	if cfg.toolResultHook != nil {
		adapter.WithToolResultHook(cfg.toolResultHook)
	}

	if cfg.sourceHook != nil {
		adapter.WithSourceHook(cfg.sourceHook)
	}

	if cfg.onFinish != nil {
		fn := cfg.onFinish
		adapter.WithOnFinish(func(text, _ string) {
			fn(text)
		})
	}

	return adapter.Stream(sr.Events(), w)
}
