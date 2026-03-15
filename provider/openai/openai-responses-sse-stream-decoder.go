package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/ssgohq/ai-go/ai"
)

// responsesChunk represents a single SSE event from the OpenAI Responses API stream.
// Only the fields needed for stream normalization are decoded.
type responsesChunk struct {
	Type string `json:"type"`

	// response.created / response.completed
	Response *struct {
		ID     string `json:"id"`
		Status string `json:"status"`
		Usage  *struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	} `json:"response"`

	// response.output_text.delta
	Delta string `json:"delta"`

	// response.output_item.added (web search action / function call)
	Item *struct {
		Type      string `json:"type"`
		ID        string `json:"id"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"item"`

	// response.function_call_arguments.delta
	// response.function_call_arguments.done
	ItemID      string `json:"item_id"`
	OutputIndex int    `json:"output_index"`

	// response.reasoning_summary_text.delta
	// (same Delta field, distinguished by Type)

	// response.web_search_call.action.sources (when include requested)
	Sources []struct {
		Type  string `json:"type"`
		ID    string `json:"id"`
		URL   string `json:"url"`
		Title string `json:"title"`
	} `json:"sources"`

	// Error event
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// decodeResponsesSSEStream reads OpenAI Responses API SSE lines and emits
// normalized ai.StreamEvents onto ch. Closes ch when done or on error.
// encodingWarnings are merged onto the finish event so callers see them in the
// GenerateTextResult.Warnings field without a separate event.
func decodeResponsesSSEStream(ctx context.Context, body io.ReadCloser, ch chan<- ai.StreamEvent, encodingWarnings ...ai.Warning) {
	defer close(ch)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	// Track active function calls for accumulating argument deltas.
	type pendingCall struct {
		id   string
		name string
		args strings.Builder
	}
	callsByItemID := make(map[string]*pendingCall)
	var callOrder []string // item_id order for index assignment

	// Track last response ID for provider metadata.
	var responseID string

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			ch <- ai.StreamEvent{Type: ai.StreamEventError, Error: ctx.Err()}
			return
		default:
		}

		line := scanner.Text()

		// SSE format: lines beginning with "data: " carry JSON payloads.
		// Lines beginning with "event: " carry event type names (ignored; we use .type inside JSON).
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return
		}

		var chunk responsesChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			ch <- ai.StreamEvent{
				Type:  ai.StreamEventError,
				Error: fmt.Errorf("openai: unmarshal responses chunk: %w", err),
			}
			return
		}

		switch chunk.Type {
		case "response.created":
			if chunk.Response != nil {
				responseID = chunk.Response.ID
			}

		case "response.completed":
			if chunk.Response != nil {
				if chunk.Response.ID != "" {
					responseID = chunk.Response.ID
				}
				var usage *ai.Usage
				if u := chunk.Response.Usage; u != nil {
					usage = &ai.Usage{
						PromptTokens:     u.InputTokens,
						CompletionTokens: u.OutputTokens,
						TotalTokens:      u.TotalTokens,
					}
				}
				providerMeta := map[string]any{
					"openai": map[string]any{
						"responseId": responseID,
					},
				}
				if usage != nil {
					ch <- ai.StreamEvent{Type: ai.StreamEventUsage, Usage: usage}
				}
				ch <- ai.StreamEvent{
					Type:             ai.StreamEventFinish,
					FinishReason:     mapResponsesFinishReason(chunk.Response.Status, len(callsByItemID) > 0),
					RawFinishReason:  chunk.Response.Status,
					ProviderMetadata: providerMeta,
					Warnings:         encodingWarnings,
				}
			}

		case "response.failed", "response.cancelled", "response.incomplete":
			status := chunk.Type
			ch <- ai.StreamEvent{
				Type:            ai.StreamEventFinish,
				FinishReason:    mapResponsesFinishReason(status, false),
				RawFinishReason: status,
			}
			return

		case "error":
			if chunk.Error != nil {
				ch <- ai.StreamEvent{
					Type:  ai.StreamEventError,
					Error: fmt.Errorf("openai: %s: %s", chunk.Error.Code, chunk.Error.Message),
				}
			}
			return

		case "response.output_text.delta":
			if chunk.Delta != "" {
				ch <- ai.StreamEvent{Type: ai.StreamEventTextDelta, TextDelta: chunk.Delta}
			}

		case "response.reasoning_summary_text.delta":
			if chunk.Delta != "" {
				ch <- ai.StreamEvent{Type: ai.StreamEventReasoningDelta, TextDelta: chunk.Delta}
			}

		case "response.output_item.added":
			if chunk.Item == nil {
				continue
			}
			switch chunk.Item.Type {
			case "function_call":
				itemID := chunk.Item.ID
				pc := &pendingCall{id: chunk.Item.CallID, name: chunk.Item.Name}
				callsByItemID[itemID] = pc
				callOrder = append(callOrder, itemID)
				idx := len(callOrder) - 1
				ch <- ai.StreamEvent{
					Type:          ai.StreamEventToolCallDelta,
					ToolCallIndex: idx,
					ToolCallID:    chunk.Item.CallID,
					ToolCallName:  chunk.Item.Name,
				}
			}

		case "response.function_call_arguments.delta":
			if chunk.Delta == "" || chunk.ItemID == "" {
				continue
			}
			pc, ok := callsByItemID[chunk.ItemID]
			if !ok {
				continue
			}
			pc.args.WriteString(chunk.Delta)
			idx := indexOfItemID(callOrder, chunk.ItemID)
			ch <- ai.StreamEvent{
				Type:              ai.StreamEventToolCallDelta,
				ToolCallIndex:     idx,
				ToolCallID:        pc.id,
				ToolCallName:      pc.name,
				ToolCallArgsDelta: chunk.Delta,
			}

		case "response.web_search_call.sources":
			for _, s := range chunk.Sources {
				ch <- ai.StreamEvent{
					Type: ai.StreamEventSource,
					Source: &ai.Source{
						SourceType: "url",
						ID:         s.ID,
						URL:        s.URL,
						Title:      s.Title,
					},
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		ch <- ai.StreamEvent{
			Type:  ai.StreamEventError,
			Error: fmt.Errorf("openai: read stream: %w", err),
		}
	}
}

func indexOfItemID(order []string, id string) int {
	for i, v := range order {
		if v == id {
			return i
		}
	}
	return 0
}

func mapResponsesFinishReason(status string, hasFunctionCall bool) ai.FinishReason {
	switch status {
	case "completed":
		if hasFunctionCall {
			return ai.FinishReasonToolCalls
		}
		return ai.FinishReasonStop
	case "max_output_tokens":
		return ai.FinishReasonLength
	case "content_filter":
		return ai.FinishReasonContentFilter
	default:
		if hasFunctionCall {
			return ai.FinishReasonToolCalls
		}
		return ai.FinishReasonUnknown
	}
}
