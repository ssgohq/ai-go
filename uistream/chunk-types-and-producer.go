package uistream

import (
	"encoding/json"

	"github.com/open-ai-sdk/ai-go/internal/engine"
)

// Chunk is a typed UI stream chunk that can be serialized to different transports.
type Chunk struct {
	Type   string         // chunk type constant (ChunkTextDelta, etc.)
	Fields map[string]any // chunk-specific payload fields
}

// ChunkStream is a stream of typed chunks produced from engine events.
// Drain Chunks before calling FullText.
type ChunkStream struct {
	// Chunks is the output channel; it is closed when the producer is done.
	Chunks <-chan Chunk

	done chan struct{}
	text string
}

// FullText blocks until the Chunks channel is fully drained and returns the
// accumulated assistant text.
func (cs *ChunkStream) FullText() string {
	<-cs.done
	return cs.text
}

// ChunkProducer translates engine.StepEvents into a channel of typed Chunks.
// It holds the same per-stream state as the former Adapter internals.
type ChunkProducer struct {
	msgID string

	// per-step state — reset on each StepStart
	textBlockID      string
	textBlockCount   int
	textStarted      bool
	reasoningStarted bool
	toolInputStarted map[string]bool
	toolArgsAccum    map[string]string

	// lastFinishReason stores the finish reason from the most recent StepEventStepEnd.
	lastFinishReason string
	// lastThoughtSignature stores the most recent thought signature from a reasoning delta.
	lastThoughtSignature string
}

// NewChunkProducer creates a ChunkProducer with the given message ID.
func NewChunkProducer(msgID string) *ChunkProducer {
	return &ChunkProducer{
		msgID:            msgID,
		toolInputStarted: make(map[string]bool),
		toolArgsAccum:    make(map[string]string),
	}
}

// Produce starts consuming events from ch, emitting Chunks to the returned
// ChunkStream. The returned ChunkStream.Chunks channel is closed when ch is
// exhausted or an error event is received.
//
// Produce is designed for single use per ChunkProducer instance.
func (cp *ChunkProducer) Produce(ch <-chan engine.StepEvent) *ChunkStream {
	out := make(chan Chunk, 64)
	cs := &ChunkStream{
		Chunks: out,
		done:   make(chan struct{}),
	}

	go func() {
		defer close(out)
		defer close(cs.done)

		// Emit the stream-start chunk first.
		out <- Chunk{Type: ChunkStart, Fields: map[string]any{"messageId": cp.msgID}}

		for ev := range ch {
			if ev.Type == engine.StepEventError {
				msg := "stream error"
				if ev.Error != nil {
					msg = "stream error: " + ev.Error.Error()
				}
				out <- Chunk{Type: ChunkError, Fields: map[string]any{"errorText": msg}}
				return
			}
			chunks, delta := cp.translateEvent(ev)
			for _, c := range chunks {
				out <- c
			}
			cs.text += delta
		}
	}()

	return cs
}

// translateEvent converts a single StepEvent into zero or more Chunks plus any
// text delta accumulated.
func (cp *ChunkProducer) translateEvent(ev engine.StepEvent) ([]Chunk, string) {
	switch ev.Type {
	case engine.StepEventStepStart:
		return cp.chunksStepStart(), ""
	case engine.StepEventTextDelta:
		return cp.chunksTextDelta(ev)
	case engine.StepEventReasoningDelta:
		return cp.chunksReasoningDelta(ev), ""
	case engine.StepEventToolCallStart:
		return cp.chunksToolCallStart(ev), ""
	case engine.StepEventToolCallDelta:
		return cp.chunksToolCallDelta(ev), ""
	case engine.StepEventToolResult:
		return cp.chunksToolResult(ev), ""
	case engine.StepEventSource:
		return cp.chunksSource(ev), ""
	case engine.StepEventStepEnd:
		cp.lastFinishReason = string(ev.FinishReason)
		return cp.chunksStepEnd(), ""
	case engine.StepEventDone:
		fields := map[string]any{}
		if cp.lastFinishReason != "" {
			fields["finishReason"] = cp.lastFinishReason
		}
		return []Chunk{
			{Type: ChunkFinish, Fields: fields},
		}, ""
	}
	return nil, ""
}

func (cp *ChunkProducer) chunksStepStart() []Chunk {
	cp.textBlockCount++
	cp.textBlockID = blockID(cp.textBlockCount)
	cp.textStarted = false
	cp.reasoningStarted = false
	cp.lastThoughtSignature = ""
	cp.toolInputStarted = make(map[string]bool)
	cp.toolArgsAccum = make(map[string]string)
	return []Chunk{{Type: ChunkStartStep, Fields: nil}}
}

func (cp *ChunkProducer) chunksTextDelta(ev engine.StepEvent) ([]Chunk, string) {
	var out []Chunk
	if !cp.textStarted {
		out = append(out, Chunk{Type: ChunkTextStart, Fields: map[string]any{"id": cp.textBlockID}})
		cp.textStarted = true
	}
	fields := map[string]any{
		"id":    cp.textBlockID,
		"delta": ev.TextDelta,
	}
	out = append(out, Chunk{Type: ChunkTextDelta, Fields: withProviderMetadata(fields, ev.ProviderMetadata)})
	return out, ev.TextDelta
}

func (cp *ChunkProducer) chunksReasoningDelta(ev engine.StepEvent) []Chunk {
	var out []Chunk
	if !cp.reasoningStarted {
		out = append(out, Chunk{Type: ChunkReasoningStart, Fields: map[string]any{"id": cp.textBlockID}})
		cp.reasoningStarted = true
	}
	if ev.ThoughtSignature != "" {
		cp.lastThoughtSignature = ev.ThoughtSignature
	}
	fields := map[string]any{
		"id":    cp.textBlockID,
		"delta": ev.ReasoningDelta,
	}
	out = append(out, Chunk{Type: ChunkReasoningDelta, Fields: withProviderMetadata(fields, ev.ProviderMetadata)})
	return out
}

func (cp *ChunkProducer) chunksToolCallStart(ev engine.StepEvent) []Chunk {
	tcID := ev.ToolCallID
	if tcID == "" {
		return nil
	}
	cp.toolInputStarted[tcID] = true
	cp.toolArgsAccum[tcID] = ev.ToolCallArgsDelta

	out := []Chunk{{Type: ChunkToolInputStart, Fields: map[string]any{
		"toolCallId": tcID,
		"toolName":   ev.ToolCallName,
	}}}
	if ev.ToolCallArgsDelta != "" {
		out = append(out, Chunk{Type: ChunkToolInputDelta, Fields: map[string]any{
			"toolCallId":     tcID,
			"inputTextDelta": ev.ToolCallArgsDelta,
		}})
	}
	return out
}

func (cp *ChunkProducer) chunksToolCallDelta(ev engine.StepEvent) []Chunk {
	tcID := ev.ToolCallID
	if !cp.toolInputStarted[tcID] || ev.ToolCallArgsDelta == "" {
		return nil
	}
	existing := cp.toolArgsAccum[tcID]
	if isValidJSON(existing) {
		return nil
	}
	cp.toolArgsAccum[tcID] += ev.ToolCallArgsDelta
	return []Chunk{{Type: ChunkToolInputDelta, Fields: map[string]any{
		"toolCallId":     tcID,
		"inputTextDelta": ev.ToolCallArgsDelta,
	}}}
}

func (cp *ChunkProducer) chunksToolResult(ev engine.StepEvent) []Chunk {
	if ev.ToolResult == nil {
		return nil
	}
	tr := ev.ToolResult

	var parsedArgs any
	if err := json.Unmarshal([]byte(tr.Args), &parsedArgs); err != nil {
		parsedArgs = map[string]string{"raw": tr.Args}
	}

	var parsedOutput any
	if err := json.Unmarshal([]byte(tr.Output), &parsedOutput); err != nil {
		parsedOutput = map[string]string{"result": tr.Output}
	}

	inputFields := withProviderMetadata(map[string]any{
		"toolCallId": tr.ID,
		"toolName":   tr.Name,
		"input":      parsedArgs,
	}, ev.ProviderMetadata)
	outputFields := withProviderMetadata(map[string]any{
		"toolCallId": tr.ID,
		"output":     parsedOutput,
	}, ev.ProviderMetadata)

	return []Chunk{
		{Type: ChunkToolInputAvailable, Fields: inputFields},
		{Type: ChunkToolOutputAvailable, Fields: outputFields},
	}
}

func (cp *ChunkProducer) chunksSource(ev engine.StepEvent) []Chunk {
	if ev.Source == nil || ev.Source.URL == "" {
		return nil
	}
	return []Chunk{{Type: ChunkSourceURL, Fields: map[string]any{
		"sourceId": ev.Source.ID,
		"url":      ev.Source.URL,
		"title":    ev.Source.Title,
	}}}
}

func (cp *ChunkProducer) chunksStepEnd() []Chunk {
	var out []Chunk
	if cp.textStarted {
		out = append(out, Chunk{Type: ChunkTextEnd, Fields: map[string]any{"id": cp.textBlockID}})
	}
	if cp.reasoningStarted {
		reasoningEndFields := map[string]any{"id": cp.textBlockID}
		if cp.lastThoughtSignature != "" {
			reasoningEndFields["signature"] = cp.lastThoughtSignature
		}
		out = append(out, Chunk{Type: ChunkReasoningEnd, Fields: reasoningEndFields})
	}
	out = append(out, Chunk{Type: ChunkFinishStep, Fields: nil})
	return out
}

// blockID returns a text block identifier for step n.
func blockID(n int) string {
	return "text_" + itoa(n)
}

// itoa is a minimal int-to-string to avoid importing strconv for this one use.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	digits := make([]byte, 0, 10)
	for n > 0 {
		digits = append(digits, byte('0'+n%10))
		n /= 10
	}
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}
	return string(digits)
}

func isValidJSON(s string) bool {
	return json.Valid([]byte(s))
}

// withProviderMetadata attaches providerMetadata to a Fields map when pm is non-nil.
// Returns the (possibly newly allocated) map.
func withProviderMetadata(fields map[string]any, pm map[string]any) map[string]any {
	if pm == nil {
		return fields
	}
	if fields == nil {
		fields = make(map[string]any)
	}
	fields["providerMetadata"] = pm
	return fields
}

// MergeChunks merges multiple chunk channels into one output channel.
// Each source is drained concurrently; order within a single source is preserved
// but interleaving between sources is non-deterministic.
func MergeChunks(sources ...<-chan Chunk) <-chan Chunk {
	out := make(chan Chunk, 64)
	if len(sources) == 0 {
		close(out)
		return out
	}

	remaining := make(chan struct{}, len(sources))
	for _, src := range sources {
		src := src
		go func() {
			for c := range src {
				out <- c
			}
			remaining <- struct{}{}
		}()
	}

	go func() {
		for i := 0; i < len(sources); i++ {
			<-remaining
		}
		close(out)
	}()

	return out
}
