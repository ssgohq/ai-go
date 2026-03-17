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
		filteredCh = intercepted
	}

	producer := NewChunkProducer(msgID)
	cs := producer.Produce(filteredCh)

	// If no metadata callback, return chunks directly.
	if opts.MessageMetadata == nil {
		return cs.Chunks
	}

	// Wrap to attach metadata to the finish chunk.
	out := make(chan Chunk, 64)
	go func() {
		defer close(out)
		var lastFinishReason string
		for c := range cs.Chunks {
			if c.Type == ChunkFinish {
				if fr, ok := c.Fields["finishReason"].(string); ok {
					lastFinishReason = fr
				}
				metadata := opts.MessageMetadata(MessageMetadataInfo{
					FinishReason: lastFinishReason,
					Usage:        &totalUsage,
				})
				if metadata != nil {
					if c.Fields == nil {
						c.Fields = make(map[string]any)
					}
					c.Fields["messageMetadata"] = metadata
				}
			}
			out <- c
		}
	}()
	return out
}
