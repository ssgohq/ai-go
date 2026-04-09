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
		Type string `json:"type"` // "text", "tool_use", "thinking"
		ID   string `json:"id,omitempty"`
		Name string `json:"name,omitempty"`
		Text string `json:"text,omitempty"`
	} `json:"content_block"`
}

type sseContentBlockDelta struct {
	Index int `json:"index"`
	Delta struct {
		Type        string `json:"type"` // "text_delta", "input_json_delta", "thinking_delta"
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

// decodeSSEStream reads Anthropic SSE events and emits normalized ai.StreamEvents.
func decodeSSEStream(ctx context.Context, body io.ReadCloser, out chan<- ai.StreamEvent) {
	defer close(out)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)

	var eventType string
	// Track active content blocks for tool call index mapping
	type blockState struct {
		index int
		typ   string
		id    string
		name  string
	}
	blocks := make(map[int]*blockState)

	for scanner.Scan() {
		if ctx.Err() != nil {
			out <- ai.StreamEvent{Type: ai.StreamEventError, Error: ctx.Err()}
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

		switch eventType {
		case eventMessageStart:
			var msg sseMessageStart
			if json.Unmarshal([]byte(data), &msg) == nil {
				u := msg.Message.Usage
				out <- ai.StreamEvent{
					Type: ai.StreamEventUsage,
					Usage: &ai.Usage{
						PromptTokens:     u.InputTokens,
						CompletionTokens: u.OutputTokens,
						CacheReadTokens:  u.CacheReadInputTokens,
						CacheWriteTokens: u.CacheCreationInputTokens,
					},
				}
			}

		case eventContentBlockStart:
			var block sseContentBlockStart
			if json.Unmarshal([]byte(data), &block) == nil {
				bs := &blockState{
					index: block.Index,
					typ:   block.ContentBlock.Type,
					id:    block.ContentBlock.ID,
					name:  block.ContentBlock.Name,
				}
				blocks[block.Index] = bs

				if block.ContentBlock.Type == "tool_use" {
					out <- ai.StreamEvent{
						Type:          ai.StreamEventToolCallDelta,
						ToolCallIndex: block.Index,
						ToolCallID:    block.ContentBlock.ID,
						ToolCallName:  block.ContentBlock.Name,
					}
				}
			}

		case eventContentBlockDelta:
			var delta sseContentBlockDelta
			if json.Unmarshal([]byte(data), &delta) == nil {
				bs := blocks[delta.Index]
				switch delta.Delta.Type {
				case "text_delta":
					out <- ai.StreamEvent{
						Type:      ai.StreamEventTextDelta,
						TextDelta: delta.Delta.Text,
					}
				case "input_json_delta":
					if bs != nil {
						out <- ai.StreamEvent{
							Type:              ai.StreamEventToolCallDelta,
							ToolCallIndex:     delta.Index,
							ToolCallID:        bs.id,
							ToolCallName:      bs.name,
							ToolCallArgsDelta: delta.Delta.PartialJSON,
						}
					}
				case "thinking_delta":
					out <- ai.StreamEvent{
						Type:      ai.StreamEventReasoningDelta,
						TextDelta: delta.Delta.Thinking,
					}
				}
			}

		case eventMessageDelta:
			var msg sseMessageDelta
			if json.Unmarshal([]byte(data), &msg) == nil {
				reason := mapStopReason(msg.Delta.StopReason)
				out <- ai.StreamEvent{
					Type:            ai.StreamEventFinish,
					FinishReason:    reason,
					RawFinishReason: msg.Delta.StopReason,
				}
				if msg.Usage.OutputTokens > 0 {
					out <- ai.StreamEvent{
						Type: ai.StreamEventUsage,
						Usage: &ai.Usage{
							CompletionTokens: msg.Usage.OutputTokens,
						},
					}
				}
			}

		case eventError:
			var errMsg sseError
			if json.Unmarshal([]byte(data), &errMsg) == nil {
				out <- ai.StreamEvent{
					Type:  ai.StreamEventError,
					Error: fmt.Errorf("anthropic: %s: %s", errMsg.Error.Type, errMsg.Error.Message),
				}
				return
			}

		case eventPing, eventContentBlockStop, eventMessageStop:
			// No-op
		}
	}

	if err := scanner.Err(); err != nil {
		out <- ai.StreamEvent{Type: ai.StreamEventError, Error: fmt.Errorf("anthropic: read stream: %w", err)}
	}
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
