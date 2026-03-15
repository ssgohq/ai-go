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
	textBlockID    string
	textBlockCount int
	textStarted    bool
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
		switch ev.Type {
		case engine.StepEventStepStart:
			a.textBlockCount++
			a.textBlockID = fmt.Sprintf("text_%d", a.textBlockCount)
			a.textStarted = false
			a.reasoningStarted = false
			a.toolInputStarted = make(map[string]bool)
			a.toolArgsAccum = make(map[string]string)
			a.writeSSE(w, map[string]any{"type": ChunkStartStep})

		case engine.StepEventTextDelta:
			fullText += ev.TextDelta
			if !a.textStarted {
				a.writeSSE(w, map[string]any{"type": ChunkTextStart, "id": a.textBlockID})
				a.textStarted = true
			}
			a.writeSSE(w, map[string]any{
				"type":  ChunkTextDelta,
				"id":    a.textBlockID,
				"delta": ev.TextDelta,
			})

		case engine.StepEventReasoningDelta:
			if !a.reasoningStarted {
				a.writeSSE(w, map[string]any{"type": ChunkReasoningStart, "id": a.textBlockID})
				a.reasoningStarted = true
			}
			a.writeSSE(w, map[string]any{
				"type":  ChunkReasoningDelta,
				"id":    a.textBlockID,
				"delta": ev.ReasoningDelta,
			})

		case engine.StepEventToolCallStart:
			tcID := ev.ToolCallID
			if tcID == "" {
				break
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

		case engine.StepEventToolCallDelta:
			tcID := ev.ToolCallID
			if !a.toolInputStarted[tcID] || ev.ToolCallArgsDelta == "" {
				break
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

		case engine.StepEventToolResult:
			if ev.ToolResult == nil {
				break
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

		case engine.StepEventStepEnd:
			if a.textStarted {
				a.writeSSE(w, map[string]any{"type": ChunkTextEnd, "id": a.textBlockID})
			}
			if a.reasoningStarted {
				a.writeSSE(w, map[string]any{"type": ChunkReasoningEnd, "id": a.textBlockID})
			}
			a.writeSSE(w, map[string]any{"type": ChunkFinishStep})

		case engine.StepEventDone:
			a.writeSSE(w, map[string]any{"type": ChunkFinish})
			fmt.Fprintf(w, "data: [DONE]\n\n")

		case engine.StepEventError:
			a.writeSSE(w, map[string]any{
				"type":      ChunkError,
				"errorText": fmt.Sprintf("stream error: %v", ev.Error),
			})
			return fullText
		}
	}

	return fullText
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
