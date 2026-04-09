package anthropic

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/open-ai-sdk/ai-go/ai"
)

// SSE event types from Anthropic's streaming API.
const (
	eventMessageStart      = "message_start"
	eventContentBlockStart = "content_block_start"
	eventContentBlockDelta = "content_block_delta"
	eventContentBlockStop  = "content_block_stop"
	eventMessageDelta      = "message_delta"
	eventMessageStop       = "message_stop"
	eventPing              = "ping"
	eventError             = "error"
)

type sseMessageStart struct {
	Message struct {
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		} `json:"usage"`
	} `json:"message"`
}

type sseContentBlockStart struct {
	Index        int `json:"index"`
	ContentBlock struct {
		Type string `json:"type"`
		ID   string `json:"id,omitempty"`
		Name string `json:"name,omitempty"`
		Text string `json:"text,omitempty"`
	} `json:"content_block"`
}

type sseContentBlockDelta struct {
	Index int `json:"index"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text,omitempty"`
		PartialJSON string `json:"partial_json,omitempty"`
		Thinking    string `json:"thinking,omitempty"`
	} `json:"delta"`
}

type sseMessageDelta struct {
	Delta struct {
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type sseError struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// blockState tracks an active content block during SSE decoding.
type blockState struct {
	index int
	typ   string
	id    string
	name  string
}

// decodeSSEStream reads Anthropic SSE events and emits normalized ai.StreamEvents.
func decodeSSEStream(
	ctx context.Context,
	body io.ReadCloser,
	out chan<- ai.StreamEvent,
) {
	defer close(out)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)

	send := func(ev ai.StreamEvent) bool {
		select {
		case out <- ev:
			return true
		case <-ctx.Done():
			return false
		}
	}

	var eventType string
	blocks := make(map[int]*blockState)

	for scanner.Scan() {
		if ctx.Err() != nil {
			send(ai.StreamEvent{
				Type:  ai.StreamEventError,
				Error: ctx.Err(),
			})
			return
		}

		line := scanner.Text()
		if line == "" {
			eventType = ""
			continue
		}

		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if !dispatchSSEEvent(eventType, data, blocks, send) {
			return
		}
	}

	if err := scanner.Err(); err != nil {
		send(ai.StreamEvent{
			Type:  ai.StreamEventError,
			Error: fmt.Errorf("anthropic: read stream: %w", err),
		})
	}
}

// dispatchSSEEvent handles a single SSE data payload by event type.
// Returns false if the caller should stop (channel closed or terminal error).
func dispatchSSEEvent(
	eventType, data string,
	blocks map[int]*blockState,
	send func(ai.StreamEvent) bool,
) bool {
	switch eventType {
	case eventMessageStart:
		return handleMessageStart(data, send)
	case eventContentBlockStart:
		return handleContentBlockStart(data, blocks, send)
	case eventContentBlockDelta:
		return handleContentBlockDelta(data, blocks, send)
	case eventMessageDelta:
		return handleMessageDelta(data, send)
	case eventError:
		return handleError(data, send)
	}
	return true
}

func handleMessageStart(
	data string,
	send func(ai.StreamEvent) bool,
) bool {
	var msg sseMessageStart
	if json.Unmarshal([]byte(data), &msg) == nil {
		u := msg.Message.Usage
		return send(ai.StreamEvent{
			Type: ai.StreamEventUsage,
			Usage: &ai.Usage{
				PromptTokens:     u.InputTokens,
				CompletionTokens: u.OutputTokens,
				CacheReadTokens:  u.CacheReadInputTokens,
				CacheWriteTokens: u.CacheCreationInputTokens,
			},
		})
	}
	return true
}

func handleContentBlockStart(
	data string,
	blocks map[int]*blockState,
	send func(ai.StreamEvent) bool,
) bool {
	var block sseContentBlockStart
	if json.Unmarshal([]byte(data), &block) == nil {
		blocks[block.Index] = &blockState{
			index: block.Index,
			typ:   block.ContentBlock.Type,
			id:    block.ContentBlock.ID,
			name:  block.ContentBlock.Name,
		}
		if block.ContentBlock.Type == "tool_use" {
			return send(ai.StreamEvent{
				Type:          ai.StreamEventToolCallDelta,
				ToolCallIndex: block.Index,
				ToolCallID:    block.ContentBlock.ID,
				ToolCallName:  block.ContentBlock.Name,
			})
		}
	}
	return true
}

func handleContentBlockDelta(
	data string,
	blocks map[int]*blockState,
	send func(ai.StreamEvent) bool,
) bool {
	var delta sseContentBlockDelta
	if json.Unmarshal([]byte(data), &delta) != nil {
		return true
	}
	bs := blocks[delta.Index]
	switch delta.Delta.Type {
	case "text_delta":
		return send(ai.StreamEvent{
			Type:      ai.StreamEventTextDelta,
			TextDelta: delta.Delta.Text,
		})
	case "input_json_delta":
		if bs != nil {
			return send(ai.StreamEvent{
				Type:              ai.StreamEventToolCallDelta,
				ToolCallIndex:     delta.Index,
				ToolCallID:        bs.id,
				ToolCallName:      bs.name,
				ToolCallArgsDelta: delta.Delta.PartialJSON,
			})
		}
	case "thinking_delta":
		return send(ai.StreamEvent{
			Type:      ai.StreamEventReasoningDelta,
			TextDelta: delta.Delta.Thinking,
		})
	}
	return true
}

func handleMessageDelta(
	data string,
	send func(ai.StreamEvent) bool,
) bool {
	var msg sseMessageDelta
	if json.Unmarshal([]byte(data), &msg) != nil {
		return true
	}
	// Emit usage before finish so consumers don't miss the final token count.
	if msg.Usage.OutputTokens > 0 {
		if !send(ai.StreamEvent{
			Type: ai.StreamEventUsage,
			Usage: &ai.Usage{
				CompletionTokens: msg.Usage.OutputTokens,
			},
		}) {
			return false
		}
	}
	return send(ai.StreamEvent{
		Type:            ai.StreamEventFinish,
		FinishReason:    mapStopReason(msg.Delta.StopReason),
		RawFinishReason: msg.Delta.StopReason,
	})
}

func handleError(
	data string,
	send func(ai.StreamEvent) bool,
) bool {
	var errMsg sseError
	if json.Unmarshal([]byte(data), &errMsg) == nil {
		send(ai.StreamEvent{
			Type: ai.StreamEventError,
			Error: fmt.Errorf(
				"anthropic: %s: %s",
				errMsg.Error.Type, errMsg.Error.Message,
			),
		})
		return false
	}
	return true
}

func mapStopReason(reason string) ai.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return ai.FinishReasonStop
	case "tool_use":
		return ai.FinishReasonToolCalls
	case "max_tokens":
		return ai.FinishReasonLength
	default:
		return ai.FinishReasonUnknown
	}
}
