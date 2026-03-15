package uistream

import (
	"encoding/json"
	"fmt"
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

// Stream consumes events from ch and writes SSE lines to w.
// It returns the full concatenated assistant text for persistence.
func (a *Adapter) Stream(ch <-chan engine.StepEvent, w io.Writer) string {
	a.writeSSE(w, map[string]any{"type": ChunkStart, "messageId": a.msgID})

	var fullText string

	for ev := range ch {
		if ev.Type == engine.StepEventError {
			a.handleError(w, ev)
			return fullText
		}
		fullText += a.handleEvent(w, ev)
	}

	return fullText
}

// handleEvent dispatches a single StepEvent and returns any text delta accumulated.
func (a *Adapter) handleEvent(w io.Writer, ev engine.StepEvent) string {
	switch ev.Type {
	case engine.StepEventStepStart:
		a.handleStepStart(w)
	case engine.StepEventTextDelta:
		return a.handleTextDelta(w, ev)
	case engine.StepEventReasoningDelta:
		a.handleReasoningDelta(w, ev)
	case engine.StepEventToolCallStart:
		a.handleToolCallStart(w, ev)
	case engine.StepEventToolCallDelta:
		a.handleToolCallDelta(w, ev)
	case engine.StepEventToolResult:
		a.handleToolResult(w, ev)
	case engine.StepEventStepEnd:
		a.handleStepEnd(w)
	case engine.StepEventDone:
		a.writeSSE(w, map[string]any{"type": ChunkFinish})
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}
	return ""
}

func (a *Adapter) handleStepStart(w io.Writer) {
	a.textBlockCount++
	a.textBlockID = fmt.Sprintf("text_%d", a.textBlockCount)
	a.textStarted = false
	a.reasoningStarted = false
	a.toolInputStarted = make(map[string]bool)
	a.toolArgsAccum = make(map[string]string)
	a.writeSSE(w, map[string]any{"type": ChunkStartStep})
}

func (a *Adapter) handleTextDelta(w io.Writer, ev engine.StepEvent) string {
	if !a.textStarted {
		a.writeSSE(w, map[string]any{"type": ChunkTextStart, "id": a.textBlockID})
		a.textStarted = true
	}
	a.writeSSE(w, map[string]any{
		"type":  ChunkTextDelta,
		"id":    a.textBlockID,
		"delta": ev.TextDelta,
	})
	return ev.TextDelta
}

func (a *Adapter) handleReasoningDelta(w io.Writer, ev engine.StepEvent) {
	if !a.reasoningStarted {
		a.writeSSE(w, map[string]any{"type": ChunkReasoningStart, "id": a.textBlockID})
		a.reasoningStarted = true
	}
	a.writeSSE(w, map[string]any{
		"type":  ChunkReasoningDelta,
		"id":    a.textBlockID,
		"delta": ev.ReasoningDelta,
	})
}

func (a *Adapter) handleToolCallStart(w io.Writer, ev engine.StepEvent) {
	tcID := ev.ToolCallID
	if tcID == "" {
		return
	}
	a.toolInputStarted[tcID] = true
	a.toolArgsAccum[tcID] = ev.ToolCallArgsDelta
	a.writeSSE(w, map[string]any{
		"type":       ChunkToolInputStart,
		"toolCallId": tcID,
		"toolName":   ev.ToolCallName,
	})
	if ev.ToolCallArgsDelta != "" {
		a.writeSSE(w, map[string]any{
			"type":           ChunkToolInputDelta,
			"toolCallId":     tcID,
			"inputTextDelta": ev.ToolCallArgsDelta,
		})
	}
}

func (a *Adapter) handleToolCallDelta(w io.Writer, ev engine.StepEvent) {
	tcID := ev.ToolCallID
	if !a.toolInputStarted[tcID] || ev.ToolCallArgsDelta == "" {
		return
	}
	existing := a.toolArgsAccum[tcID]
	if !isValidJSON(existing) {
		a.toolArgsAccum[tcID] += ev.ToolCallArgsDelta
		a.writeSSE(w, map[string]any{
			"type":           ChunkToolInputDelta,
			"toolCallId":     tcID,
			"inputTextDelta": ev.ToolCallArgsDelta,
		})
	}
}

func (a *Adapter) handleToolResult(w io.Writer, ev engine.StepEvent) {
	if ev.ToolResult == nil {
		return
	}
	tr := ev.ToolResult

	var parsedArgs any
	if err := json.Unmarshal([]byte(tr.Args), &parsedArgs); err != nil {
		parsedArgs = map[string]string{"raw": tr.Args}
	}
	a.writeSSE(w, map[string]any{
		"type":       ChunkToolInputAvailable,
		"toolCallId": tr.ID,
		"toolName":   tr.Name,
		"input":      parsedArgs,
	})

	var parsedOutput any
	if err := json.Unmarshal([]byte(tr.Output), &parsedOutput); err != nil {
		parsedOutput = map[string]string{"result": tr.Output}
	}
	a.writeSSE(w, map[string]any{
		"type":       ChunkToolOutputAvailable,
		"toolCallId": tr.ID,
		"output":     parsedOutput,
	})
}

func (a *Adapter) handleStepEnd(w io.Writer) {
	if a.textStarted {
		a.writeSSE(w, map[string]any{"type": ChunkTextEnd, "id": a.textBlockID})
	}
	if a.reasoningStarted {
		a.writeSSE(w, map[string]any{"type": ChunkReasoningEnd, "id": a.textBlockID})
	}
	a.writeSSE(w, map[string]any{"type": ChunkFinishStep})
}

func (a *Adapter) handleError(w io.Writer, ev engine.StepEvent) {
	a.writeSSE(w, map[string]any{
		"type":      ChunkError,
		"errorText": fmt.Sprintf("stream error: %v", ev.Error),
	})
}

func (a *Adapter) writeSSE(w io.Writer, payload map[string]any) {
	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", b)
}

func isValidJSON(s string) bool {
	return json.Valid([]byte(s))
}
