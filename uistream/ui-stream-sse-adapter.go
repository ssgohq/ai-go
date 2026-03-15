package uistream

import (
	"encoding/json"
	"io"

	"github.com/ssgohq/ai-go/internal/engine"
)

// Chunk is a single UI message stream chunk ready for SSE serialization.
type Chunk struct {
	Type string
	Data map[string]any
}

// Adapter translates a channel of engine.StepEvents into UI message stream chunks.
// It is transport-agnostic: callers can write to an http.ResponseWriter, a buffer, etc.
//
// For custom data-* chunks or source chunks between or after stream events,
// obtain the Writer via adapter.Writer(w) and call WriteData / WriteSource directly.
type Adapter struct {
	msgID string

	// per-step state
	textBlockID      string
	textBlockCount   int
	textStarted      bool
	reasoningStarted bool

	toolInputStarted map[string]bool
	toolArgsAccum    map[string]string
}

// NewAdapter creates an Adapter with a fixed message ID.
func NewAdapter(msgID string) *Adapter {
	return &Adapter{
		msgID:            msgID,
		toolInputStarted: make(map[string]bool),
		toolArgsAccum:    make(map[string]string),
	}
}

// Writer returns a direct-write Writer bound to w for emitting custom chunks
// alongside a stream (e.g. WriteData, WriteSource).
func (a *Adapter) Writer(w io.Writer) *Writer {
	return NewWriter(w)
}

// Stream consumes events from ch and writes SSE lines to w.
// It returns the full concatenated assistant text for persistence.
func (a *Adapter) Stream(ch <-chan engine.StepEvent, w io.Writer) string {
	wr := NewWriter(w)
	wr.WriteStart(a.msgID)

	var fullText string

	for ev := range ch {
		if ev.Type == engine.StepEventError {
			a.handleError(wr, ev)
			return fullText
		}
		fullText += a.handleEvent(wr, ev)
	}

	return fullText
}

// handleEvent dispatches a single StepEvent and returns any text delta accumulated.
func (a *Adapter) handleEvent(wr *Writer, ev engine.StepEvent) string {
	switch ev.Type {
	case engine.StepEventStepStart:
		a.handleStepStart(wr)
	case engine.StepEventTextDelta:
		return a.handleTextDelta(wr, ev)
	case engine.StepEventReasoningDelta:
		a.handleReasoningDelta(wr, ev)
	case engine.StepEventToolCallStart:
		a.handleToolCallStart(wr, ev)
	case engine.StepEventToolCallDelta:
		a.handleToolCallDelta(wr, ev)
	case engine.StepEventToolResult:
		a.handleToolResult(wr, ev)
	case engine.StepEventStepEnd:
		a.handleStepEnd(wr)
	case engine.StepEventDone:
		wr.WriteFinish()
	}
	return ""
}

func (a *Adapter) handleStepStart(wr *Writer) {
	a.textBlockCount++
	a.textBlockID = blockID(a.textBlockCount)
	a.textStarted = false
	a.reasoningStarted = false
	a.toolInputStarted = make(map[string]bool)
	a.toolArgsAccum = make(map[string]string)
	wr.WriteChunk(ChunkStartStep, nil)
}

func (a *Adapter) handleTextDelta(wr *Writer, ev engine.StepEvent) string {
	if !a.textStarted {
		wr.WriteChunk(ChunkTextStart, map[string]any{"id": a.textBlockID})
		a.textStarted = true
	}
	wr.WriteChunk(ChunkTextDelta, map[string]any{
		"id":    a.textBlockID,
		"delta": ev.TextDelta,
	})
	return ev.TextDelta
}

func (a *Adapter) handleReasoningDelta(wr *Writer, ev engine.StepEvent) {
	if !a.reasoningStarted {
		wr.WriteChunk(ChunkReasoningStart, map[string]any{"id": a.textBlockID})
		a.reasoningStarted = true
	}
	wr.WriteChunk(ChunkReasoningDelta, map[string]any{
		"id":    a.textBlockID,
		"delta": ev.ReasoningDelta,
	})
}

func (a *Adapter) handleToolCallStart(wr *Writer, ev engine.StepEvent) {
	tcID := ev.ToolCallID
	if tcID == "" {
		return
	}
	a.toolInputStarted[tcID] = true
	a.toolArgsAccum[tcID] = ev.ToolCallArgsDelta
	wr.WriteChunk(ChunkToolInputStart, map[string]any{
		"toolCallId": tcID,
		"toolName":   ev.ToolCallName,
	})
	if ev.ToolCallArgsDelta != "" {
		wr.WriteChunk(ChunkToolInputDelta, map[string]any{
			"toolCallId":     tcID,
			"inputTextDelta": ev.ToolCallArgsDelta,
		})
	}
}

func (a *Adapter) handleToolCallDelta(wr *Writer, ev engine.StepEvent) {
	tcID := ev.ToolCallID
	if !a.toolInputStarted[tcID] || ev.ToolCallArgsDelta == "" {
		return
	}
	existing := a.toolArgsAccum[tcID]
	if !isValidJSON(existing) {
		a.toolArgsAccum[tcID] += ev.ToolCallArgsDelta
		wr.WriteChunk(ChunkToolInputDelta, map[string]any{
			"toolCallId":     tcID,
			"inputTextDelta": ev.ToolCallArgsDelta,
		})
	}
}

func (a *Adapter) handleToolResult(wr *Writer, ev engine.StepEvent) {
	if ev.ToolResult == nil {
		return
	}
	tr := ev.ToolResult

	var parsedArgs any
	if err := json.Unmarshal([]byte(tr.Args), &parsedArgs); err != nil {
		parsedArgs = map[string]string{"raw": tr.Args}
	}
	wr.WriteChunk(ChunkToolInputAvailable, map[string]any{
		"toolCallId": tr.ID,
		"toolName":   tr.Name,
		"input":      parsedArgs,
	})

	var parsedOutput any
	if err := json.Unmarshal([]byte(tr.Output), &parsedOutput); err != nil {
		parsedOutput = map[string]string{"result": tr.Output}
	}
	wr.WriteChunk(ChunkToolOutputAvailable, map[string]any{
		"toolCallId": tr.ID,
		"output":     parsedOutput,
	})
}

func (a *Adapter) handleStepEnd(wr *Writer) {
	if a.textStarted {
		wr.WriteChunk(ChunkTextEnd, map[string]any{"id": a.textBlockID})
	}
	if a.reasoningStarted {
		wr.WriteChunk(ChunkReasoningEnd, map[string]any{"id": a.textBlockID})
	}
	wr.WriteChunk(ChunkFinishStep, nil)
}

func (a *Adapter) handleError(wr *Writer, ev engine.StepEvent) {
	msg := "stream error"
	if ev.Error != nil {
		msg = "stream error: " + ev.Error.Error()
	}
	wr.WriteError(msg)
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
	// reverse
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}
	return string(digits)
}

func isValidJSON(s string) bool {
	return json.Valid([]byte(s))
}
