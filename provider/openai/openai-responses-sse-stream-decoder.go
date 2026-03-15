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

// streamState holds mutable state accumulated during SSE stream decoding.
type streamState struct {
	responseID    string
	callsByItemID map[string]*pendingCall
	callOrder     []string
}

type pendingCall struct {
	id   string
	name string
	args strings.Builder
}

// decodeResponsesSSEStream reads OpenAI Responses API SSE lines and emits
// normalized ai.StreamEvents onto ch. Closes ch when done or on error.
// encodingWarnings are merged onto the finish event so callers see them in the
// GenerateTextResult.Warnings field without a separate event.
func decodeResponsesSSEStream(
	ctx context.Context, body io.ReadCloser, ch chan<- ai.StreamEvent,
	encodingWarnings ...ai.Warning,
) {
	defer close(ch)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	state := &streamState{callsByItemID: make(map[string]*pendingCall)}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			ch <- ai.StreamEvent{Type: ai.StreamEventError, Error: ctx.Err()}
			return
		default:
		}

		line := scanner.Text()
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

		if done := dispatchChunk(chunk, state, ch, encodingWarnings); done {
			return
		}
	}

	if err := scanner.Err(); err != nil {
		ch <- ai.StreamEvent{
			Type:  ai.StreamEventError,
			Error: fmt.Errorf("openai: read stream: %w", err),
		}
	}
}

// dispatchChunk routes a single SSE chunk to the appropriate handler.
// Returns true if the stream should terminate.
func dispatchChunk(
	chunk responsesChunk,
	state *streamState,
	ch chan<- ai.StreamEvent,
	encodingWarnings []ai.Warning,
) bool {
	switch chunk.Type {
	case "response.created":
		if chunk.Response != nil {
			state.responseID = chunk.Response.ID
		}

	case "response.completed":
		handleResponseCompleted(chunk, state, ch, encodingWarnings)

	case "response.failed", "response.cancelled", "response.incomplete":
		ch <- ai.StreamEvent{
			Type:            ai.StreamEventFinish,
			FinishReason:    mapResponsesFinishReason(chunk.Type, false),
			RawFinishReason: chunk.Type,
		}
		return true

	case "error":
		if chunk.Error != nil {
			ch <- ai.StreamEvent{
				Type:  ai.StreamEventError,
				Error: fmt.Errorf("openai: %s: %s", chunk.Error.Code, chunk.Error.Message),
			}
		}
		return true

	case "response.output_text.delta":
		if chunk.Delta != "" {
			ch <- ai.StreamEvent{Type: ai.StreamEventTextDelta, TextDelta: chunk.Delta}
		}

	case "response.reasoning_summary_text.delta":
		if chunk.Delta != "" {
			ch <- ai.StreamEvent{Type: ai.StreamEventReasoningDelta, TextDelta: chunk.Delta}
		}

	case "response.output_item.added":
		handleOutputItemAdded(chunk, state, ch)

	case "response.function_call_arguments.delta":
		handleFunctionCallArgsDelta(chunk, state, ch)

	case "response.web_search_call.sources":
		for _, s := range chunk.Sources {
			ch <- ai.StreamEvent{
				Type:   ai.StreamEventSource,
				Source: &ai.Source{SourceType: "url", ID: s.ID, URL: s.URL, Title: s.Title},
			}
		}
	}
	return false
}

func handleResponseCompleted(
	chunk responsesChunk,
	state *streamState,
	ch chan<- ai.StreamEvent,
	encodingWarnings []ai.Warning,
) {
	if chunk.Response == nil {
		return
	}
	if chunk.Response.ID != "" {
		state.responseID = chunk.Response.ID
	}
	if u := chunk.Response.Usage; u != nil {
		ch <- ai.StreamEvent{Type: ai.StreamEventUsage, Usage: &ai.Usage{
			PromptTokens:     u.InputTokens,
			CompletionTokens: u.OutputTokens,
			TotalTokens:      u.TotalTokens,
		}}
	}
	ch <- ai.StreamEvent{
		Type:             ai.StreamEventFinish,
		FinishReason:     mapResponsesFinishReason(chunk.Response.Status, len(state.callsByItemID) > 0),
		RawFinishReason:  chunk.Response.Status,
		ProviderMetadata: map[string]any{"openai": map[string]any{"responseId": state.responseID}},
		Warnings:         encodingWarnings,
	}
}

func handleOutputItemAdded(chunk responsesChunk, state *streamState, ch chan<- ai.StreamEvent) {
	if chunk.Item == nil || chunk.Item.Type != "function_call" {
		return
	}
	itemID := chunk.Item.ID
	pc := &pendingCall{id: chunk.Item.CallID, name: chunk.Item.Name}
	state.callsByItemID[itemID] = pc
	state.callOrder = append(state.callOrder, itemID)
	ch <- ai.StreamEvent{
		Type:          ai.StreamEventToolCallDelta,
		ToolCallIndex: len(state.callOrder) - 1,
		ToolCallID:    chunk.Item.CallID,
		ToolCallName:  chunk.Item.Name,
	}
}

func handleFunctionCallArgsDelta(chunk responsesChunk, state *streamState, ch chan<- ai.StreamEvent) {
	if chunk.Delta == "" || chunk.ItemID == "" {
		return
	}
	pc, ok := state.callsByItemID[chunk.ItemID]
	if !ok {
		return
	}
	pc.args.WriteString(chunk.Delta)
	ch <- ai.StreamEvent{
		Type:              ai.StreamEventToolCallDelta,
		ToolCallIndex:     indexOfItemID(state.callOrder, chunk.ItemID),
		ToolCallID:        pc.id,
		ToolCallName:      pc.name,
		ToolCallArgsDelta: chunk.Delta,
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
