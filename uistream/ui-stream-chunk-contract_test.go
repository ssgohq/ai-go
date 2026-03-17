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
		ChunkSourceURL,
		ChunkSourceDocument,
		ChunkFile,
		ChunkAbort,
		ChunkMessageMetadata,
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
		ChunkSourceURL:           "source-url",
		ChunkSourceDocument:      "source-document",
		ChunkFile:                "file",
		ChunkAbort:               "abort",
		ChunkMessageMetadata:     "message-metadata",
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

// TestWriter_WriteSourceDocument_V6Fields verifies filename, data, providerMetadata appear.
func TestWriter_WriteSourceDocument_V6Fields(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteSourceDocument("src-1", "application/pdf", "My Doc", &SourceDocumentOpts{
			Filename:         "report.pdf",
			Data:             []byte("pdfcontent"),
			ProviderMetadata: map[string]any{"provider": "test"},
		})
	})
	if !strings.Contains(output, `"type":"source-document"`) {
		t.Errorf("WriteSourceDocument: expected type=source-document\ngot: %s", output)
	}
	if !strings.Contains(output, `"filename":"report.pdf"`) {
		t.Errorf("WriteSourceDocument: expected filename\ngot: %s", output)
	}
	if !strings.Contains(output, `"providerMetadata"`) {
		t.Errorf("WriteSourceDocument: expected providerMetadata\ngot: %s", output)
	}
}

// TestWriter_WriteSourceDocument_NilOpts verifies nil opts emits basic fields only.
func TestWriter_WriteSourceDocument_NilOpts(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteSourceDocument("src-2", "text/plain", "Bare Doc", nil)
	})
	if !strings.Contains(output, `"type":"source-document"`) {
		t.Errorf("WriteSourceDocument nil opts: expected type\ngot: %s", output)
	}
	if strings.Contains(output, `"filename"`) {
		t.Errorf("WriteSourceDocument nil opts: unexpected filename field\ngot: %s", output)
	}
}

// TestWriter_WriteFile_V6Fields verifies id, fileId, data, name, providerMetadata appear.
func TestWriter_WriteFile_V6Fields(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteFile("https://cdn.example.com/f.png", "image/png", &FileChunkOpts{
			ID:               "file-id-1",
			FileID:           "fid-42",
			Data:             []byte("imgdata"),
			Name:             "photo.png",
			ProviderMetadata: map[string]any{"bucket": "s3"},
		})
	})
	if !strings.Contains(output, `"type":"file"`) {
		t.Errorf("WriteFile: expected type=file\ngot: %s", output)
	}
	if !strings.Contains(output, `"id":"file-id-1"`) {
		t.Errorf("WriteFile: expected id\ngot: %s", output)
	}
	if !strings.Contains(output, `"fileId":"fid-42"`) {
		t.Errorf("WriteFile: expected fileId\ngot: %s", output)
	}
	if !strings.Contains(output, `"name":"photo.png"`) {
		t.Errorf("WriteFile: expected name\ngot: %s", output)
	}
	if !strings.Contains(output, `"providerMetadata"`) {
		t.Errorf("WriteFile: expected providerMetadata\ngot: %s", output)
	}
}

// TestWriter_WriteFile_NilOpts verifies nil opts emits only url and mediaType.
func TestWriter_WriteFile_NilOpts(t *testing.T) {
	output := captureWriterOutput(func(w *Writer) {
		w.WriteFile("https://cdn.example.com/f.png", "image/png", nil)
	})
	if !strings.Contains(output, `"type":"file"`) {
		t.Errorf("WriteFile nil opts: expected type\ngot: %s", output)
	}
	if strings.Contains(output, `"id"`) || strings.Contains(output, `"fileId"`) {
		t.Errorf("WriteFile nil opts: unexpected id/fileId fields\ngot: %s", output)
	}
}

// TestWriter_ToolChunkOpts_ProviderExecuted verifies bool fields appear in tool error/denied chunks.
func TestWriter_ToolChunkOpts_ProviderExecuted(t *testing.T) {
	boolTrue := true
	boolFalse := false

	t.Run("WriteToolInputError with opts", func(t *testing.T) {
		output := captureWriterOutput(func(w *Writer) {
			w.WriteToolInputError("tc-1", "myTool", nil, "bad input", &ToolChunkOpts{
				ProviderExecuted: &boolTrue,
				Dynamic:          &boolFalse,
				Title:            "My Tool",
			})
		})
		if !strings.Contains(output, `"providerExecuted":true`) {
			t.Errorf("WriteToolInputError opts: expected providerExecuted\ngot: %s", output)
		}
		if !strings.Contains(output, `"dynamic":false`) {
			t.Errorf("WriteToolInputError opts: expected dynamic\ngot: %s", output)
		}
		if !strings.Contains(output, `"title":"My Tool"`) {
			t.Errorf("WriteToolInputError opts: expected title\ngot: %s", output)
		}
	})

	t.Run("WriteToolOutputError with opts", func(t *testing.T) {
		output := captureWriterOutput(func(w *Writer) {
			w.WriteToolOutputError("tc-2", "exec failed", &ToolChunkOpts{
				ProviderExecuted: &boolTrue,
			})
		})
		if !strings.Contains(output, `"providerExecuted":true`) {
			t.Errorf("WriteToolOutputError opts: expected providerExecuted\ngot: %s", output)
		}
	})

	t.Run("WriteToolOutputDenied with opts", func(t *testing.T) {
		output := captureWriterOutput(func(w *Writer) {
			w.WriteToolOutputDenied("tc-3", &ToolChunkOpts{Dynamic: &boolTrue})
		})
		if !strings.Contains(output, `"dynamic":true`) {
			t.Errorf("WriteToolOutputDenied opts: expected dynamic\ngot: %s", output)
		}
	})

	t.Run("WriteToolOutputAvailable preliminary via WriteChunk", func(t *testing.T) {
		output := captureWriterOutput(func(w *Writer) {
			w.WriteChunk(ChunkToolOutputAvailable, map[string]any{
				"toolCallId":  "tc-4",
				"output":      "result",
				"preliminary": true,
			})
		})
		if !strings.Contains(output, `"preliminary":true`) {
			t.Errorf("WriteChunk tool-output-available: expected preliminary\ngot: %s", output)
		}
	})
}
