package gemini

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/open-ai-sdk/ai-go/ai"
)

// --------------------------------------------------------------------
// Native Gemini SSE response types
// --------------------------------------------------------------------

type nativeSSEChunk struct {
	Candidates    []nativeCandidate    `json:"candidates"`
	UsageMetadata *nativeUsageMetadata `json:"usageMetadata"`
	ModelVersion  string               `json:"modelVersion"`
}

type nativeCandidate struct {
	Content           *nativeCandidateContent  `json:"content"`
	FinishReason      string                   `json:"finishReason"`
	Index             int                      `json:"index"`
	GroundingMetadata *nativeGroundingMetadata `json:"groundingMetadata"`
	SafetyRatings     []any                    `json:"safetyRatings"`
}

type nativeCandidateContent struct {
	Parts []nativeResponsePart `json:"parts"`
	Role  string               `json:"role"`
}

type nativeResponsePart struct {
	Text             string             `json:"text,omitempty"`
	Thought          *bool              `json:"thought,omitempty"`
	ThoughtSignature string             `json:"thoughtSignature,omitempty"`
	FunctionCall     *nativeSSEFuncCall `json:"functionCall,omitempty"`
	InlineData       *nativeInlineData  `json:"inlineData,omitempty"`
}

type nativeSSEFuncCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

// nativeInlineData is defined in native_request_encoder.go

type nativeUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
}

type nativeGroundingMetadata struct {
	WebSearchQueries  []string               `json:"webSearchQueries"`
	GroundingChunks   []nativeGroundingChunk `json:"groundingChunks"`
	GroundingSupports []any                  `json:"groundingSupports"`
}

type nativeGroundingChunk struct {
	Web              *nativeWebChunk     `json:"web,omitempty"`
	Image            *nativeImageChunk   `json:"image,omitempty"`
	RetrievedContext *nativeRetrievedCtx `json:"retrievedContext,omitempty"`
	Maps             *nativeMapsChunk    `json:"maps,omitempty"`
}

type nativeWebChunk struct {
	URI   string `json:"uri"`
	Title string `json:"title"`
}

type nativeImageChunk struct {
	URI   string `json:"uri"`
	Title string `json:"title"`
}

type nativeRetrievedCtx struct {
	URI   string `json:"uri"`
	Title string `json:"title"`
}

type nativeMapsChunk struct {
	URI   string `json:"uri"`
	Title string `json:"title"`
}

// --------------------------------------------------------------------
// Decoder
// --------------------------------------------------------------------

// decodeNativeSSEStream reads Gemini native SSE from body and emits ai.StreamEvents onto ch.
// It closes ch when the stream ends (EOF or context cancellation).
// body is closed when decoding finishes.
func decodeNativeSSEStream(ctx context.Context, body io.ReadCloser, ch chan<- ai.StreamEvent) {
	defer close(ch)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	seen := make(map[string]bool)
	var lastGoogleMeta map[string]any
	toolCallIndex := 0

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
		if data == "" {
			continue
		}

		var chunk nativeSSEChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			ch <- ai.StreamEvent{
				Type:  ai.StreamEventError,
				Error: fmt.Errorf("gemini-native: unmarshal chunk: %w", err),
			}
			continue
		}

		emitNativeChunkEvents(chunk, ch, seen, &lastGoogleMeta, &toolCallIndex)
	}

	if err := scanner.Err(); err != nil {
		ch <- ai.StreamEvent{
			Type:  ai.StreamEventError,
			Error: fmt.Errorf("gemini-native: read stream: %w", err),
		}
	}
}

// emitNativeChunkEvents processes a single native SSE chunk and emits events in order:
// sources → content parts → usage → finish.
func emitNativeChunkEvents(
	chunk nativeSSEChunk,
	ch chan<- ai.StreamEvent,
	seen map[string]bool,
	lastGoogleMeta *map[string]any,
	toolCallIndex *int,
) {
	hasToolCalls := false

	if len(chunk.Candidates) > 0 {
		cand := chunk.Candidates[0]

		// Accumulate google metadata from grounding + safety across chunks.
		if meta := buildNativeGoogleMetadata(cand); meta != nil {
			*lastGoogleMeta = meta
		}

		// 1. Sources from grounding metadata.
		for _, src := range extractNativeGroundingSources(cand.GroundingMetadata, seen) {
			s := src
			ch <- ai.StreamEvent{Type: ai.StreamEventSource, Source: &s}
		}

		// 2. Content parts.
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				if part.FunctionCall != nil {
					hasToolCalls = true
					args := string(part.FunctionCall.Args)
					ch <- ai.StreamEvent{
						Type:              ai.StreamEventToolCallDelta,
						ToolCallIndex:     *toolCallIndex,
						ToolCallID:        fmt.Sprintf("call_%d", *toolCallIndex),
						ToolCallName:      part.FunctionCall.Name,
						ToolCallArgsDelta: args,
						ThoughtSignature:  part.ThoughtSignature,
					}
					*toolCallIndex++
					continue
				}

				if part.Thought != nil && *part.Thought {
					ch <- ai.StreamEvent{
						Type:             ai.StreamEventReasoningDelta,
						TextDelta:        part.Text,
						ThoughtSignature: part.ThoughtSignature,
					}
				} else if part.Text != "" {
					ch <- ai.StreamEvent{
						Type:             ai.StreamEventTextDelta,
						TextDelta:        part.Text,
						ThoughtSignature: part.ThoughtSignature,
					}
				}
			}
		}

		// 3. Usage (emit before finish so consumers see token counts first).
		if chunk.UsageMetadata != nil {
			ch <- ai.StreamEvent{
				Type: ai.StreamEventUsage,
				Usage: &ai.Usage{
					PromptTokens:     chunk.UsageMetadata.PromptTokenCount,
					CompletionTokens: chunk.UsageMetadata.CandidatesTokenCount,
					TotalTokens:      chunk.UsageMetadata.TotalTokenCount,
					ReasoningTokens:  chunk.UsageMetadata.ThoughtsTokenCount,
				},
			}
		}

		// 4. Finish.
		if cand.FinishReason != "" {
			reason, raw := mapNativeFinishReason(cand.FinishReason, hasToolCalls)
			var provMeta map[string]any
			if *lastGoogleMeta != nil {
				provMeta = map[string]any{"google": *lastGoogleMeta}
			}
			ch <- ai.StreamEvent{
				Type:             ai.StreamEventFinish,
				FinishReason:     reason,
				RawFinishReason:  raw,
				ProviderMetadata: provMeta,
			}
		}

		return
	}

	// No candidates — might still have usage metadata.
	if chunk.UsageMetadata != nil {
		ch <- ai.StreamEvent{
			Type: ai.StreamEventUsage,
			Usage: &ai.Usage{
				PromptTokens:     chunk.UsageMetadata.PromptTokenCount,
				CompletionTokens: chunk.UsageMetadata.CandidatesTokenCount,
				TotalTokens:      chunk.UsageMetadata.TotalTokenCount,
				ReasoningTokens:  chunk.UsageMetadata.ThoughtsTokenCount,
			},
		}
	}
}

// mapNativeFinishReason converts a Gemini native API finish reason to an ai.FinishReason.
func mapNativeFinishReason(raw string, hasToolCalls bool) (ai.FinishReason, string) {
	switch raw {
	case "STOP":
		if hasToolCalls {
			return ai.FinishReasonToolCalls, raw
		}
		return ai.FinishReasonStop, raw
	case "MAX_TOKENS":
		return ai.FinishReasonLength, raw
	case "SAFETY", "RECITATION", "BLOCKLIST":
		return ai.FinishReasonContentFilter, raw
	case "MALFORMED_FUNCTION_CALL":
		return ai.FinishReasonError, raw
	default:
		return ai.FinishReasonUnknown, raw
	}
}

// extractNativeGroundingSources extracts deduplicated ai.Source values from grounding metadata.
func extractNativeGroundingSources(gm *nativeGroundingMetadata, seen map[string]bool) []ai.Source {
	if gm == nil {
		return nil
	}
	var sources []ai.Source
	for _, gc := range gm.GroundingChunks {
		if gc.Web != nil {
			if src := extractNativeWebSource(gc.Web, seen); src != nil {
				sources = append(sources, *src)
				continue
			}
		}
		if gc.RetrievedContext != nil {
			if src := extractNativeRetrievedContextSource(gc.RetrievedContext, seen); src != nil {
				sources = append(sources, *src)
				continue
			}
		}
		if gc.Image != nil {
			if src := extractNativeImageSource(gc.Image, seen); src != nil {
				sources = append(sources, *src)
				continue
			}
		}
		if gc.Maps != nil {
			if src := extractNativeMapsSource(gc.Maps, seen); src != nil {
				sources = append(sources, *src)
			}
		}
	}
	return sources
}

func extractNativeWebSource(web *nativeWebChunk, seen map[string]bool) *ai.Source {
	if web.URI == "" {
		return nil
	}
	if seen[web.URI] {
		return nil
	}
	seen[web.URI] = true
	return &ai.Source{
		SourceType: "url",
		URL:        web.URI,
		Title:      web.Title,
	}
}

func extractNativeRetrievedContextSource(rc *nativeRetrievedCtx, seen map[string]bool) *ai.Source {
	if rc.URI == "" {
		return nil
	}
	if seen[rc.URI] {
		return nil
	}
	seen[rc.URI] = true
	return &ai.Source{
		SourceType: "retrieved-context",
		URL:        rc.URI,
		Title:      rc.Title,
	}
}

func extractNativeImageSource(img *nativeImageChunk, seen map[string]bool) *ai.Source {
	uri := img.URI
	if uri == "" {
		uri = "image-chunk"
	}
	key := "image:" + uri
	if seen[key] {
		return nil
	}
	seen[key] = true
	return &ai.Source{
		SourceType: "image",
		URL:        uri,
		Title:      img.Title,
	}
}

func extractNativeMapsSource(m *nativeMapsChunk, seen map[string]bool) *ai.Source {
	uri := m.URI
	if uri == "" {
		uri = "maps-chunk"
	}
	key := "maps:" + uri
	if seen[key] {
		return nil
	}
	seen[key] = true
	return &ai.Source{
		SourceType: "maps",
		URL:        uri,
		Title:      m.Title,
	}
}

// buildNativeGoogleMetadata assembles provider metadata from a native candidate.
func buildNativeGoogleMetadata(cand nativeCandidate) map[string]any {
	result := make(map[string]any)
	if cand.GroundingMetadata != nil {
		result["groundingMetadata"] = cand.GroundingMetadata
	}
	if cand.SafetyRatings != nil {
		result["safetyRatings"] = cand.SafetyRatings
	}
	if len(result) == 0 {
		return nil
	}
	return result
}
