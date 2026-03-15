package uistream

import (
	"bytes"
	"strings"
	"testing"

	"github.com/ssgohq/ai-go/internal/engine"
)

// TestWriter_WriteData verifies that WriteData emits a valid data-* chunk.
func TestWriter_WriteData(t *testing.T) {
	var buf bytes.Buffer
	wr := NewWriter(&buf)

	wr.WriteData("plan", map[string]any{"content": "Research AI trends"})

	out := buf.String()
	assertContains(t, out, `"type":"data-plan"`)
	assertContains(t, out, `"data"`)
	assertContains(t, out, `"content":"Research AI trends"`)
}

// TestWriter_WriteSource verifies that WriteSource emits a valid source chunk.
func TestWriter_WriteSource(t *testing.T) {
	var buf bytes.Buffer
	wr := NewWriter(&buf)

	wr.WriteSource(Source{
		ID:    "src-1",
		URL:   "https://example.com/article",
		Title: "Example Article",
	})

	out := buf.String()
	assertContains(t, out, `"type":"source"`)
	assertContains(t, out, `"url":"https://example.com/article"`)
	assertContains(t, out, `"title":"Example Article"`)
	assertContains(t, out, `"id":"src-1"`)
}

// TestWriter_WriteSources verifies that WriteSources emits a sources chunk with multiple entries.
func TestWriter_WriteSources(t *testing.T) {
	var buf bytes.Buffer
	wr := NewWriter(&buf)

	wr.WriteSources([]Source{
		{URL: "https://a.com", Title: "Site A"},
		{URL: "https://b.com", Title: "Site B"},
	})

	out := buf.String()
	assertContains(t, out, `"type":"sources"`)
	assertContains(t, out, `"https://a.com"`)
	assertContains(t, out, `"https://b.com"`)
}

// TestWriter_WriteChunk verifies arbitrary chunk emission.
func TestWriter_WriteChunk(t *testing.T) {
	var buf bytes.Buffer
	wr := NewWriter(&buf)

	wr.WriteChunk("data-step-1-start", map[string]any{"step": 1})

	out := buf.String()
	assertContains(t, out, `"type":"data-step-1-start"`)
	assertContains(t, out, `"step":1`)
}

// TestWriter_FullLifecycle verifies the start/finish/[DONE] frame.
func TestWriter_FullLifecycle(t *testing.T) {
	var buf bytes.Buffer
	wr := NewWriter(&buf)

	wr.WriteStart("msg-abc")
	wr.WriteFinish()

	out := buf.String()
	assertContains(t, out, `"type":"start"`)
	assertContains(t, out, `"messageId":"msg-abc"`)
	assertContains(t, out, `"type":"finish"`)
	assertContains(t, out, "[DONE]")
}

// TestWriter_WriteError verifies the error chunk.
func TestWriter_WriteError(t *testing.T) {
	var buf bytes.Buffer
	wr := NewWriter(&buf)
	wr.WriteError("something went wrong")

	out := buf.String()
	assertContains(t, out, `"type":"error"`)
	assertContains(t, out, `"errorText":"something went wrong"`)
}

// TestAdapterWriter_ReturnsWriter verifies Adapter.Writer() returns a usable Writer.
func TestAdapterWriter_ReturnsWriter(t *testing.T) {
	var buf bytes.Buffer
	a := NewAdapter("msg-test")

	w := a.Writer(&buf)
	if w == nil {
		t.Fatal("expected non-nil Writer")
	}
	w.WriteData("test", "value")
	assertContains(t, buf.String(), `"type":"data-test"`)
}

// TestGolden_DeepThinking_WithSourcesAndCustomData models a full deep-thinking
// chunk sequence: start, data-plan, data-steps, reasoning step, data-step-1, sources,
// text answer, data-artifacts, data-sources, data-suggested-questions, finish.
func TestGolden_DeepThinking_WithSourcesAndCustomData(t *testing.T) {
	var buf bytes.Buffer

	// The adapter handles the engine stream (reasoning + text steps).
	a := NewAdapter("msg-deep-1")
	wr := a.Writer(&buf)

	// 1. Stream start
	wr.WriteStart("msg-deep-1")

	// 2. Custom data-plan (emitted before engine stream, by app layer)
	wr.WriteData("plan", map[string]any{"content": "Researching AI trends in 2025"})

	// 3. Custom data-steps
	wr.WriteData("steps", map[string]any{
		"steps": []string{"Research information", "Analyze data", "Summarize results"},
	})

	// 4. Engine stream: reasoning step + web search tool + text answer
	engineCh := makeEvents(
		engine.StepEvent{Type: engine.StepEventStepStart, StepNumber: 0},
		engine.StepEvent{Type: engine.StepEventReasoningDelta, ReasoningDelta: "I need to find recent AI trends..."},
		engine.StepEvent{Type: engine.StepEventReasoningDelta, ReasoningDelta: " Let me search the web."},
		engine.StepEvent{
			Type:              engine.StepEventToolCallStart,
			ToolCallID:        "ws-1",
			ToolCallName:      "web_search",
			ToolCallArgsDelta: `{"query":"AI trends 2025"}`,
		},
		engine.StepEvent{
			Type: engine.StepEventToolResult,
			ToolResult: &engine.ToolResult{
				ID:     "ws-1",
				Name:   "web_search",
				Args:   `{"query":"AI trends 2025"}`,
				Output: `{"results":["GPT-5 released","Gemini 2.0 released"]}`,
			},
		},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonToolCalls},

		engine.StepEvent{Type: engine.StepEventStepStart, StepNumber: 1},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "AI in 2025 saw major releases "},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "from Google and OpenAI."},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	)

	// Merge the engine stream (without emitting its own start/finish; we drive those manually).
	// We use a helper that runs through events without re-emitting start/finish from Adapter.
	// For this test we use a separate buffer for the engine portion and merge it manually.
	var engineBuf bytes.Buffer
	engineAdapter := NewAdapter("msg-deep-1-engine")
	engineText := engineAdapter.Stream(engineCh, &engineBuf)
	_ = engineText

	// Copy engine output (minus start/finish wrapper) - we use raw merging here.
	// In real usage callers use Adapter.Stream for the engine portion then wrap with Writer for the rest.
	// For the golden test we validate each portion separately.
	engineOut := engineBuf.String()

	// 5. After engine stream: sources
	wr.WriteSources([]Source{
		{URL: "https://openai.com/gpt5", Title: "GPT-5 Released"},
		{URL: "https://deepmind.com/gemini", Title: "Gemini 2.0"},
	})

	// 6. Custom data chunks for app-specific artifacts and questions
	wr.WriteData("artifacts", map[string]any{
		"attachments": []map[string]any{
			{"name": "report.md", "url": "https://storage.example.com/report.md"},
		},
	})
	wr.WriteData("suggested-questions", map[string]any{
		"questions": []string{
			"What are the top AI models in 2025?",
			"How does GPT-5 compare to Gemini 2.0?",
			"What are the key AI trends for 2026?",
		},
	})

	// 7. Finish
	wr.WriteFinish()

	// --- assertions on Writer output ---
	writerOut := buf.String()

	assertContains(t, writerOut, `"type":"start"`)
	assertContains(t, writerOut, `"messageId":"msg-deep-1"`)
	assertContains(t, writerOut, `"type":"data-plan"`)
	assertContains(t, writerOut, `"Researching AI trends in 2025"`)
	assertContains(t, writerOut, `"type":"data-steps"`)
	assertContains(t, writerOut, `"Research information"`)
	assertContains(t, writerOut, `"type":"sources"`)
	assertContains(t, writerOut, `"https://openai.com/gpt5"`)
	assertContains(t, writerOut, `"GPT-5 Released"`)
	assertContains(t, writerOut, `"type":"data-artifacts"`)
	assertContains(t, writerOut, `"report.md"`)
	assertContains(t, writerOut, `"type":"data-suggested-questions"`)
	assertContains(t, writerOut, `"What are the top AI models in 2025?"`)
	assertContains(t, writerOut, `"type":"finish"`)
	assertContains(t, writerOut, "[DONE]")

	// --- assertions on engine stream output ---
	assertContains(t, engineOut, `"type":"start"`)
	assertContains(t, engineOut, `"type":"start-step"`)
	assertContains(t, engineOut, `"type":"reasoning-start"`)
	assertContains(t, engineOut, `"type":"reasoning-delta"`)
	assertContains(t, engineOut, `"I need to find recent AI trends..."`)
	assertContains(t, engineOut, `"type":"tool-input-start"`)
	assertContains(t, engineOut, `"toolName":"web_search"`)
	assertContains(t, engineOut, `"type":"tool-output-available"`)
	assertContains(t, engineOut, `"type":"finish-step"`)
	assertContains(t, engineOut, `"type":"text-start"`)
	assertContains(t, engineOut, `"AI in 2025 saw major releases "`)
	assertContains(t, engineOut, `"type":"text-end"`)
	assertContains(t, engineOut, `"type":"finish"`)
}

// TestGolden_DefaultChat models the default chat chunk sequence:
// start, text-start, text-delta(s), text-end, finish.
func TestGolden_DefaultChat(t *testing.T) {
	var buf bytes.Buffer
	a := NewAdapter("msg-default-1")

	text := a.Stream(makeEvents(
		engine.StepEvent{Type: engine.StepEventStepStart, StepNumber: 0},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "Hello! "},
		engine.StepEvent{Type: engine.StepEventTextDelta, TextDelta: "How can I help you today?"},
		engine.StepEvent{Type: engine.StepEventStepEnd, FinishReason: engine.FinishReasonStop},
		engine.StepEvent{Type: engine.StepEventDone},
	), &buf)

	out := buf.String()
	assertContains(t, out, `"type":"start"`)
	assertContains(t, out, `"messageId":"msg-default-1"`)
	assertContains(t, out, `"type":"text-start"`)
	assertContains(t, out, `"delta":"Hello! "`)
	assertContains(t, out, `"delta":"How can I help you today?"`)
	assertContains(t, out, `"type":"text-end"`)
	assertContains(t, out, `"type":"finish"`)
	assertContains(t, out, "[DONE]")

	if text != "Hello! How can I help you today?" {
		t.Errorf("unexpected full text: %q", text)
	}
}

// TestGolden_DataChunksOrderPreserved verifies that data-* chunks appear in emission order.
func TestGolden_DataChunksOrderPreserved(t *testing.T) {
	var buf bytes.Buffer
	wr := NewWriter(&buf)

	wr.WriteData("step-1-start", nil)
	wr.WriteData("step-1", map[string]any{"type": "reasoning", "content": "step 1 detail"})
	wr.WriteData("step-1-end", nil)
	wr.WriteData("step-2-start", nil)
	wr.WriteData("step-2", map[string]any{"content": "step 2 detail"})
	wr.WriteData("step-2-end", nil)

	out := buf.String()
	lines := strings.Split(strings.TrimSpace(out), "\n\n")

	// 6 data lines expected (each WriteData call produces one SSE line pair)
	if len(lines) < 6 {
		t.Errorf("expected at least 6 SSE frames, got %d\noutput:\n%s", len(lines), out)
	}

	expected := []string{
		`"type":"data-step-1-start"`,
		`"type":"data-step-1"`,
		`"type":"data-step-1-end"`,
		`"type":"data-step-2-start"`,
		`"type":"data-step-2"`,
		`"type":"data-step-2-end"`,
	}
	for i, want := range expected {
		if i >= len(lines) {
			t.Errorf("missing frame %d: want %q", i, want)
			continue
		}
		if !strings.Contains(lines[i], want) {
			t.Errorf("frame %d: want %q in %q", i, want, lines[i])
		}
	}
}
