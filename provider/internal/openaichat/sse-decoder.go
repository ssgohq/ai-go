package openaichat

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/open-ai-sdk/ai-go/ai"
)

// StreamChunk mirrors the OpenAI chat completions SSE chunk structure.
// Provider-specific delta fields (e.g. Gemini thought flags) are included optionally.
type StreamChunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			Thought          *bool  `json:"thought,omitempty"`
			ThoughtSignature string `json:"thought_signature,omitempty"`
			ToolCalls        []struct {
				Index        int    `json:"index"`
				ID           string `json:"id"`
				ExtraContent *struct {
					Google struct {
						ThoughtSignature string `json:"thought_signature"`
					} `json:"google"`
				} `json:"extra_content"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens       int `json:"prompt_tokens"`
		CompletionTokens   int `json:"completion_tokens"`
		TotalTokens        int `json:"total_tokens"`
		ThoughtsTokenCount int `json:"thoughts_token_count,omitempty"`
	} `json:"usage"`
	// ProviderMetadata holds provider-specific metadata from the response (e.g. Gemini groundingMetadata).
	ProviderMetadata map[string]any `json:"provider_metadata,omitempty"`
}

// SSEDecodeParams holds configuration for DecodeSSEStream.
type SSEDecodeParams struct {
	// ProviderName is used in error messages (e.g. "gemini", "openai").
	ProviderName string
	// MetadataExtractor is an optional hook to populate ProviderMetadata on finish events.
	MetadataExtractor func(chunk StreamChunk) map[string]any
	// SourceExtractor is an optional hook to extract ai.Source events from a chunk.
	// Called for every chunk; returned sources are emitted before text/tool deltas.
	SourceExtractor func(chunk StreamChunk) []ai.Source
}

// DecodeSSEStream reads SSE lines from body and emits normalized ai.StreamEvents onto ch.
// It closes ch when done or on error.
func DecodeSSEStream(
	ctx context.Context,
	body io.ReadCloser,
	ch chan<- ai.StreamEvent,
	params SSEDecodeParams,
) {
	defer close(ch)
	defer body.Close()

	providerName := params.ProviderName
	if providerName == "" {
		providerName = "openaichat"
	}

	scanner := bufio.NewScanner(body)
	// Allow up to 1 MB per SSE line (tool schemas can be large).
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

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
			ch <- ai.StreamEvent{
				Type:            ai.StreamEventFinish,
				FinishReason:    ai.FinishReasonStop,
				RawFinishReason: "stop",
			}
			return
		}

		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			ch <- ai.StreamEvent{
				Type:  ai.StreamEventError,
				Error: fmt.Errorf("%s: unmarshal chunk: %w", providerName, err),
			}
			return
		}

		emitChunkEvents(chunk, ch, params.MetadataExtractor, params.SourceExtractor)
	}

	if err := scanner.Err(); err != nil {
		ch <- ai.StreamEvent{
			Type:  ai.StreamEventError,
			Error: fmt.Errorf("%s: read stream: %w", providerName, err),
		}
	}
}

func emitChunkEvents(
	chunk StreamChunk,
	ch chan<- ai.StreamEvent,
	metaExtractor func(StreamChunk) map[string]any,
	sourceExtractor func(StreamChunk) []ai.Source,
) {
	// Emit usage when present (may arrive on a chunk with empty choices).
	if chunk.Usage != nil {
		ch <- ai.StreamEvent{
			Type: ai.StreamEventUsage,
			Usage: &ai.Usage{
				PromptTokens:     chunk.Usage.PromptTokens,
				CompletionTokens: chunk.Usage.CompletionTokens,
				TotalTokens:      chunk.Usage.TotalTokens,
				ReasoningTokens:  chunk.Usage.ThoughtsTokenCount,
			},
		}
	}

	// Emit sources extracted from this chunk (e.g. Gemini grounding chunks).
	if sourceExtractor != nil {
		for _, src := range sourceExtractor(chunk) {
			s := src // copy for pointer safety
			ch <- ai.StreamEvent{Type: ai.StreamEventSource, Source: &s}
		}
	}

	if len(chunk.Choices) == 0 {
		return
	}
	choice := chunk.Choices[0]

	// Emit finish reason if present.
	if choice.FinishReason != "" && choice.FinishReason != "null" {
		var meta map[string]any
		if metaExtractor != nil {
			meta = metaExtractor(chunk)
		}
		ch <- ai.StreamEvent{
			Type:             ai.StreamEventFinish,
			FinishReason:     MapFinishReason(choice.FinishReason),
			RawFinishReason:  choice.FinishReason,
			ProviderMetadata: meta,
		}
	}

	// Text or reasoning delta.
	if choice.Delta.Content != "" || choice.Delta.ThoughtSignature != "" {
		if choice.Delta.Thought != nil && *choice.Delta.Thought {
			ch <- ai.StreamEvent{
				Type:             ai.StreamEventReasoningDelta,
				TextDelta:        choice.Delta.Content,
				ThoughtSignature: choice.Delta.ThoughtSignature,
			}
		} else if choice.Delta.Content != "" {
			ch <- ai.StreamEvent{
				Type:             ai.StreamEventTextDelta,
				TextDelta:        choice.Delta.Content,
				ThoughtSignature: choice.Delta.ThoughtSignature,
			}
		}
	}

	// Tool call argument deltas.
	for _, tc := range choice.Delta.ToolCalls {
		var sig string
		if tc.ExtraContent != nil {
			sig = tc.ExtraContent.Google.ThoughtSignature
		}
		ch <- ai.StreamEvent{
			Type:              ai.StreamEventToolCallDelta,
			ToolCallIndex:     tc.Index,
			ToolCallID:        tc.ID,
			ToolCallName:      tc.Function.Name,
			ToolCallArgsDelta: tc.Function.Arguments,
			ThoughtSignature:  sig,
		}
	}
}
