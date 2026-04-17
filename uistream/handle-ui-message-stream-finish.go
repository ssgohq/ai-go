package uistream

import "encoding/json"

// StepFinishInfo is the argument passed to the OnStepFinish callback.
type StepFinishInfo struct {
	// IsContinuation is true when appending to an existing assistant message.
	IsContinuation bool
	// ResponseMessage is a snapshot of the reconstructed assistant message at step boundary.
	ResponseMessage StreamingUIMessage
}

// FinishInfo is the argument passed to the OnFinish callback.
type FinishInfo struct {
	// IsAborted is true if an abort chunk was seen.
	IsAborted bool
	// IsContinuation is true when appending to an existing assistant message.
	IsContinuation bool
	// ResponseMessage is the final reconstructed assistant message.
	ResponseMessage StreamingUIMessage
	// FinishReason from the stream (e.g. "stop", "tool-calls", "error").
	FinishReason string
}

// HandleUIMessageStreamFinishOptions configures HandleUIMessageStreamFinish.
type HandleUIMessageStreamFinishOptions struct {
	// MessageID to use for the response message. If the start chunk already
	// contains a messageId, the start chunk's value takes precedence.
	MessageID string

	// LastAssistantMessage, if non-nil and role=="assistant", indicates that
	// the stream is continuing a previous assistant message. Parts from the
	// existing message are preserved and the message ID is inherited.
	LastAssistantMessage *StreamingUIMessage

	// OnStepFinish is called on every finish-step chunk with a snapshot of
	// the reconstructed assistant message.
	OnStepFinish func(info StepFinishInfo)

	// OnFinish is called once when the stream completes (flush) or is
	// cancelled.
	OnFinish func(info FinishInfo)

	// OnError is called when an error chunk is received.
	OnError func(err error)
}

// HandleUIMessageStreamFinish wraps a chunk channel with stateful processing
// and lifecycle callbacks. It injects a messageId into the start chunk if
// missing, drives processUIMessageStream for state accumulation, and fires
// OnStepFinish on every finish-step and OnFinish when the stream is fully
// consumed.
//
// This is the Go equivalent of AI SDK Node's handleUIMessageStreamFinish.
func HandleUIMessageStreamFinish(chunks <-chan Chunk, opts HandleUIMessageStreamFinishOptions) <-chan Chunk {
	// Determine effective message ID and whether this is a continuation.
	messageID := opts.MessageID
	var lastMsg *StreamingUIMessage
	if opts.LastAssistantMessage != nil && opts.LastAssistantMessage.Role == "assistant" {
		lastMsg = opts.LastAssistantMessage
		messageID = lastMsg.ID // use existing assistant message ID
	}

	// If no callbacks, pass through with messageId injection only.
	if opts.OnFinish == nil && opts.OnStepFinish == nil {
		return injectMessageID(chunks, messageID)
	}

	// Create state for accumulating the assistant message.
	state := NewStreamingUIMessageState(messageID, lastMsg)

	// Inject messageId, process state, and fire callbacks in a single
	// goroutine to guarantee callback ordering matches chunk ordering.
	out := make(chan Chunk, 64)
	go func() {
		defer close(out)

		isAborted := false
		finishCalled := false

		isContinuation := func() bool {
			return lastMsg != nil && state.Message.ID == lastMsg.ID
		}

		callOnFinish := func() {
			if finishCalled || opts.OnFinish == nil {
				return
			}
			finishCalled = true
			opts.OnFinish(FinishInfo{
				IsAborted:       isAborted,
				IsContinuation:  isContinuation(),
				ResponseMessage: snapshotMessage(state.Message),
				FinishReason:    state.FinishReason,
			})
		}

		for c := range chunks {
			// Phase 1: inject messageId into start chunk.
			if c.Type == ChunkStart && messageID != "" {
				existing, ok := c.Fields["messageId"].(string)
				if !ok || existing == "" {
					if c.Fields == nil {
						c.Fields = make(map[string]any)
					}
					c.Fields["messageId"] = messageID
				}
			}

			// Phase 2: process chunk into state.
			processChunkIntoState(c, state)

			// Phase 3: fire callbacks.
			if c.Type == ChunkAbort {
				isAborted = true
			}

			if c.Type == ChunkFinishStep && opts.OnStepFinish != nil {
				opts.OnStepFinish(StepFinishInfo{
					IsContinuation:  isContinuation(),
					ResponseMessage: snapshotMessage(state.Message),
				})
			}

			out <- c
		}

		// Stream fully consumed — call onFinish.
		callOnFinish()
	}()

	return out
}

// injectMessageID wraps a chunk channel, injecting a messageId into the start
// chunk when the chunk doesn't already have one.
func injectMessageID(chunks <-chan Chunk, messageID string) <-chan Chunk {
	if messageID == "" {
		return chunks
	}
	out := make(chan Chunk, 64)
	go func() {
		defer close(out)
		for c := range chunks {
			if c.Type == ChunkStart {
				existing, ok := c.Fields["messageId"].(string)
				if !ok || existing == "" {
					if c.Fields == nil {
						c.Fields = make(map[string]any)
					}
					c.Fields["messageId"] = messageID
				}
			}
			out <- c
		}
	}()
	return out
}

// snapshotMessage returns a shallow copy of the message with a cloned parts slice.
func snapshotMessage(msg StreamingUIMessage) StreamingUIMessage {
	clone := msg
	clone.Parts = make([]UIMessagePart, len(msg.Parts))
	for i, p := range msg.Parts {
		cp := make(UIMessagePart, len(p))
		for k, v := range p {
			cp[k] = v
		}
		clone.Parts[i] = cp
	}
	if msg.Metadata != nil {
		clone.Metadata = make(json.RawMessage, len(msg.Metadata))
		copy(clone.Metadata, msg.Metadata)
	}
	return clone
}
