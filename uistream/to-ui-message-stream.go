package uistream

import (
	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// ToUIStreamOptions configures the ToUIMessageStream bridge.
type ToUIStreamOptions struct {
	// SendReasoning forwards reasoning-start/delta/end chunks when true.
	SendReasoning bool
	// SendSources forwards source chunks when true.
	SendSources bool
	// MessageMetadata is called to attach metadata to the finish chunk.
	// If nil, no metadata is attached.
	MessageMetadata func(info MessageMetadataInfo) map[string]any

	// SendStart controls whether the start chunk is emitted (default: nil = true).
	// Set to boolPtr(false) when merging into an outer stream that already emitted start.
	SendStart *bool
	// SendFinish controls whether the finish chunk is emitted (default: nil = true).
	// Set to boolPtr(false) when the outer stream manages the finish lifecycle.
	SendFinish *bool
}

// boolVal returns the dereferenced value of a *bool, defaulting to def if nil.
func boolVal(b *bool, def bool) bool {
	if b == nil {
		return def
	}
	return *b
}

// MessageMetadataInfo provides context for the MessageMetadata callback.
type MessageMetadataInfo struct {
	FinishReason string
	Usage        *UsageInfo
}

// UsageInfo holds token count info for metadata callbacks.
type UsageInfo struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	ReasoningTokens  int
	CacheReadTokens  int
	CacheWriteTokens int
}

// ToUIMessageStream converts events from a StreamEventer into a channel of typed Chunks.
// It bridges StreamResult.Events() → UI protocol chunks with configurable options.
// The returned channel is closed when the stream completes.
//
// This is the Go equivalent of AI SDK Node's result.toUIMessageStream({ sendReasoning, messageMetadata }).
func ToUIMessageStream(sr StreamEventer, msgID string, opts ToUIStreamOptions) <-chan Chunk {
	sr.DrainUnused()

	eventCh := sr.Events()

	// Determine whether we need to intercept events.
	needIntercept := !opts.SendReasoning || !opts.SendSources || opts.MessageMetadata != nil

	filteredCh := eventCh
	var totalUsage UsageInfo

	if needIntercept {
		filteredCh = interceptEvents(eventCh, opts, &totalUsage)
	}

	producer := NewChunkProducer(msgID)
	cs := producer.Produce(filteredCh)

	sendStart := boolVal(opts.SendStart, true)
	sendFinish := boolVal(opts.SendFinish, true)
	needLifecycleFilter := !sendStart || !sendFinish

	// If no metadata callback and no lifecycle filtering, return chunks directly.
	if opts.MessageMetadata == nil && !needLifecycleFilter {
		return cs.Chunks
	}

	// Wrap to attach metadata and/or filter lifecycle chunks.
	return wrapChunksWithMetadata(cs.Chunks, opts, sendStart, sendFinish, &totalUsage)
}

// interceptEvents filters and tracks usage from the raw engine event stream.
// It writes accumulated usage into totalUsage (updated concurrently by the goroutine).
func interceptEvents(
	eventCh <-chan engine.StepEvent,
	opts ToUIStreamOptions,
	totalUsage *UsageInfo,
) <-chan engine.StepEvent {
	intercepted := make(chan engine.StepEvent, 64)

	go func() {
		defer close(intercepted)
		for ev := range eventCh {
			// Track usage for metadata.
			if ev.Type == engine.StepEventUsage && ev.Usage != nil {
				totalUsage.PromptTokens += ev.Usage.PromptTokens
				totalUsage.CompletionTokens += ev.Usage.CompletionTokens
				totalUsage.TotalTokens += ev.Usage.TotalTokens
				totalUsage.ReasoningTokens += ev.Usage.ReasoningTokens
				totalUsage.CacheReadTokens += ev.Usage.CacheReadTokens
				totalUsage.CacheWriteTokens += ev.Usage.CacheWriteTokens
			}
			// Filter reasoning events.
			if !opts.SendReasoning && ev.Type == engine.StepEventReasoningDelta {
				continue
			}
			// Filter source events.
			if !opts.SendSources && ev.Type == engine.StepEventSource {
				continue
			}
			intercepted <- ev
		}
	}()

	return intercepted
}

// wrapChunksWithMetadata wraps the chunk stream to attach metadata and/or filter lifecycle chunks.
func wrapChunksWithMetadata(
	chunks <-chan Chunk,
	opts ToUIStreamOptions,
	sendStart, sendFinish bool,
	totalUsage *UsageInfo,
) <-chan Chunk {
	out := make(chan Chunk, 64)
	go func() {
		defer close(out)
		var lastFinishReason string
		for c := range chunks {
			if !sendStart && c.Type == ChunkStart {
				continue
			}
			if !sendFinish && c.Type == ChunkFinish {
				continue
			}
			if c.Type == ChunkFinish && opts.MessageMetadata != nil {
				c = attachMessageMetadata(c, opts.MessageMetadata, lastFinishReason, totalUsage)
			}
			if c.Type == ChunkFinish {
				if fr, ok := c.Fields["finishReason"].(string); ok {
					lastFinishReason = fr
				}
			}
			out <- c
		}
	}()
	return out
}

// attachMessageMetadata calls the metadata callback and injects the result into the finish chunk.
func attachMessageMetadata(
	c Chunk,
	metaFn func(MessageMetadataInfo) map[string]any,
	finishReason string,
	usage *UsageInfo,
) Chunk {
	if fr, ok := c.Fields["finishReason"].(string); ok && fr != "" {
		finishReason = fr
	}
	metadata := metaFn(MessageMetadataInfo{
		FinishReason: finishReason,
		Usage:        usage,
	})
	if metadata != nil {
		if c.Fields == nil {
			c.Fields = make(map[string]any)
		}
		c.Fields["messageMetadata"] = metadata
	}
	return c
}
