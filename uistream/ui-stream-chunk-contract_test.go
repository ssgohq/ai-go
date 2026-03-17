package uistream

import (
	"bytes"
	"strings"
	"testing"
)

// captureWriterOutput creates a Writer backed by an in-memory buffer, runs fn
// with it, and returns the output as a string.
func captureWriterOutput(fn func(w *Writer)) string {
	var buf bytes.Buffer
	fn(NewWriter(&buf))
	return buf.String()
}

// TestChunkConstants_FrozenContractCoverage verifies every chunk type defined in
// the frozen contract (history/swift-ui-ai-sdk-mvp/contract-chat-request-and-stream.md)
// has a corresponding constant in this package.
//
// If a constant is missing, the SSE adapter cannot emit the required chunk type
// without magic strings, which would cause contract drift.
func TestChunkConstants_FrozenContractCoverage(t *testing.T) {
	required := []string{
		ChunkStart,
		ChunkStartStep,
		ChunkTextStart,
		ChunkTextDelta,
		ChunkTextEnd,
		ChunkReasoningStart,
		ChunkReasoningDelta,
		ChunkReasoningEnd,
		ChunkToolInputStart,
		ChunkToolInputDelta,
		ChunkToolInputAvailable,
		ChunkToolOutputAvailable,
		ChunkToolInputError,
		ChunkToolOutputError,
		ChunkToolOutputDenied,
		ChunkToolApprovalRequest,
		ChunkFinishStep,
		ChunkFinish,
		ChunkError,
		ChunkSource,
		ChunkSources,
	}

	// Verify each constant has the expected wire string value.
	expected := map[string]string{
		ChunkStart:               "start",
		ChunkStartStep:           "start-step",
		ChunkTextStart:           "text-start",
		ChunkTextDelta:           "text-delta",
		ChunkTextEnd:             "text-end",
		ChunkReasoningStart:      "reasoning-start",
		ChunkReasoningDelta:      "reasoning-delta",
		ChunkReasoningEnd:        "reasoning-end",
		ChunkToolInputStart:      "tool-input-start",
		ChunkToolInputDelta:      "tool-input-delta",
		ChunkToolInputAvailable:  "tool-input-available",
		ChunkToolOutputAvailable: "tool-output-available",
		ChunkToolInputError:      "tool-input-error",
		ChunkToolOutputError:     "tool-output-error",
		ChunkToolOutputDenied:    "tool-output-denied",
		ChunkToolApprovalRequest: "tool-approval-request",
		ChunkFinishStep:          "finish-step",
		ChunkFinish:              "finish",
		ChunkError:               "error",
		ChunkSource:              "source",
		ChunkSources:             "sources",
	}

	for _, c := range required {
		if c == "" {
			t.Error("found empty chunk constant — constant is undefined or empty string")
		}
		if want, ok := expected[c]; ok {
			if c != want {
				t.Errorf("chunk constant %q does not match expected wire value %q", c, want)
			}
		}
	}
}

// TestWriter_ChunkTypeInSSEPayload verifies that WriteChunk embeds the type
// field correctly in the SSE payload for each contract chunk type.
func TestWriter_ChunkTypeInSSEPayload(t *testing.T) {
	types := []string{
		ChunkStart,
		ChunkStartStep,
		ChunkTextStart,
		ChunkTextDelta,
		ChunkTextEnd,
		ChunkReasoningStart,
		ChunkReasoningDelta,
		ChunkReasoningEnd,
		ChunkToolInputStart,
		ChunkToolInputDelta,
		ChunkToolInputAvailable,
		ChunkToolOutputAvailable,
		ChunkToolInputError,
		ChunkToolOutputError,
		ChunkToolOutputDenied,
		ChunkToolApprovalRequest,
		ChunkFinishStep,
		ChunkFinish,
		ChunkError,
		ChunkSource,
		ChunkSources,
	}

	for _, typ := range types {
		output := captureWriterOutput(func(w *Writer) {
			w.WriteChunk(typ, nil)
		})
		want := `"type":"` + typ + `"`
		if !strings.Contains(output, want) {
			t.Errorf("WriteChunk(%q): SSE output missing %q\ngot: %s", typ, want, output)
		}
		if !strings.HasPrefix(output, "data: ") {
			t.Errorf("WriteChunk(%q): expected SSE prefix 'data: ', got: %s", typ, output)
		}
	}
}

// TestWriter_WriteData_PrefixConvention verifies that WriteData always produces
// a "data-<name>" type in the SSE payload.
func TestWriter_WriteData_PrefixConvention(t *testing.T) {
	cases := []struct {
		name    string
		payload any
	}{
		{"usage", map[string]int{"promptTokens": 10, "completionTokens": 5}},
		{"document-references", []map[string]string{{"id": "doc-1", "title": "Doc"}}},
		{"plan", map[string]string{"content": "step 1"}},
		{"steps", map[string][]string{"steps": {"a", "b"}}},
		{"suggested-questions", map[string][]string{"questions": {"Q1", "Q2"}}},
		{"structured-output", map[string]string{"answer": "42"}},
	}

	for _, tc := range cases {
		output := captureWriterOutput(func(w *Writer) {
			w.WriteData(tc.name, tc.payload)
		})
		want := `"type":"data-` + tc.name + `"`
		if !strings.Contains(output, want) {
			t.Errorf("WriteData(%q): missing type field %q\ngot: %s", tc.name, want, output)
		}
		if !strings.Contains(output, `"data":`) {
			t.Errorf("WriteData(%q): missing data field\ngot: %s", tc.name, output)
		}
	}
}

// TestWriter_WriteStart_MessageID verifies the start chunk includes the messageId.
func TestWriter_WriteStart_MessageID(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteStart("msg-abc-123")
	})
	if !strings.Contains(output, `"messageId":"msg-abc-123"`) {
		t.Errorf("WriteStart: expected messageId in output\ngot: %s", output)
	}
	if !strings.Contains(output, `"type":"start"`) {
		t.Errorf("WriteStart: expected type=start\ngot: %s", output)
	}
}

// TestWriter_WriteFinish_DoneTerminator verifies the finish sequence includes [DONE].
func TestWriter_WriteFinish_DoneTerminator(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteFinish()
	})
	if !strings.Contains(output, `"type":"finish"`) {
		t.Errorf("WriteFinish: expected finish chunk\ngot: %s", output)
	}
	if !strings.Contains(output, "data: [DONE]") {
		t.Errorf("WriteFinish: expected [DONE] terminator\ngot: %s", output)
	}
}

// TestWriter_WriteError_NoFinish verifies error chunk does not append a finish.
func TestWriter_WriteError_NoFinish(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteError("connection reset")
	})
	if !strings.Contains(output, `"type":"error"`) {
		t.Errorf("WriteError: expected error chunk\ngot: %s", output)
	}
	if !strings.Contains(output, `"errorText":"connection reset"`) {
		t.Errorf("WriteError: expected errorText\ngot: %s", output)
	}
	if strings.Contains(output, `"type":"finish"`) {
		t.Errorf("WriteError: must NOT emit finish chunk after error\ngot: %s", output)
	}
}

// TestWriter_WriteSource_Fields verifies source chunk shape.
func TestWriter_WriteSource_Fields(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteSource(Source{ID: "src-1", URL: "https://example.com", Title: "Example"})
	})
	if !strings.Contains(output, `"type":"source"`) {
		t.Errorf("WriteSource: expected type=source\ngot: %s", output)
	}
	if !strings.Contains(output, `"url":"https://example.com"`) {
		t.Errorf("WriteSource: expected url field\ngot: %s", output)
	}
}

// TestWriter_WriteSources_BatchShape verifies sources chunk shape.
func TestWriter_WriteSources_BatchShape(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteSources([]Source{
			{ID: "s1", URL: "https://a.com", Title: "A"},
			{ID: "s2", URL: "https://b.com", Title: "B"},
		})
	})
	if !strings.Contains(output, `"type":"sources"`) {
		t.Errorf("WriteSources: expected type=sources\ngot: %s", output)
	}
	if !strings.Contains(output, `"sources":`) {
		t.Errorf("WriteSources: expected sources array\ngot: %s", output)
	}
}
