package uistream

import (
	"encoding/json"
	"testing"
)

// chunkSliceToChan sends chunks on a channel and closes it.
func chunkSliceToChan(chunks []Chunk) <-chan Chunk {
	ch := make(chan Chunk, len(chunks))
	for _, c := range chunks {
		ch <- c
	}
	close(ch)
	return ch
}

func TestProcessUIMessageStream_TextAccumulation(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-1"}},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "Hello"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": " World"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkFinish, Fields: map[string]any{"finishReason": "stop"}},
	}

	state := NewStreamingUIMessageState("msg-1", nil)
	out := ProcessUIMessageStream(chunkSliceToChan(chunks), state)
	result := drainChunks(out)

	if len(result) != len(chunks) {
		t.Fatalf("expected %d chunks, got %d", len(chunks), len(result))
	}

	if state.Message.ID != "msg-1" {
		t.Errorf("expected message ID msg-1, got %q", state.Message.ID)
	}
	if state.Message.Role != "assistant" {
		t.Errorf("expected role assistant, got %q", state.Message.Role)
	}

	// Should have one text part with accumulated text.
	if len(state.Message.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d: %v", len(state.Message.Parts), state.Message.Parts)
	}
	part := state.Message.Parts[0]
	if part["type"] != "text" {
		t.Errorf("expected text part, got %v", part["type"])
	}
	if part["text"] != "Hello World" {
		t.Errorf("expected 'Hello World', got %q", part["text"])
	}
	if state.FinishReason != "stop" {
		t.Errorf("expected finishReason=stop, got %q", state.FinishReason)
	}
}

func TestProcessUIMessageStream_StepBoundary(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-2"}},
		// Step 1: tool call
		{Type: ChunkStartStep},
		{Type: ChunkToolInputAvailable, Fields: map[string]any{
			"toolCallId": "tc-1", "toolName": "search", "input": map[string]any{"q": "go"},
		}},
		{Type: ChunkToolOutputAvailable, Fields: map[string]any{
			"toolCallId": "tc-1", "output": "result",
		}},
		{Type: ChunkFinishStep},
		// Step 2: text
		{Type: ChunkStartStep},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "Answer"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkFinishStep},
		{Type: ChunkFinish, Fields: map[string]any{"finishReason": "stop"}},
	}

	state := NewStreamingUIMessageState("msg-2", nil)
	out := ProcessUIMessageStream(chunkSliceToChan(chunks), state)
	drainChunks(out)

	// Expected parts: step-start, tool-invocation, step-start, text
	expectedTypes := []string{"step-start", "tool-invocation", "step-start", "text"}
	if len(state.Message.Parts) != len(expectedTypes) {
		t.Fatalf("expected %d parts, got %d: %v", len(expectedTypes), len(state.Message.Parts), state.Message.Parts)
	}
	for i, expected := range expectedTypes {
		if state.Message.Parts[i]["type"] != expected {
			t.Errorf("part[%d]: expected type %q, got %q", i, expected, state.Message.Parts[i]["type"])
		}
	}

	// Active text/reasoning should be reset after finish-step.
	if len(state.ActiveTextParts) != 0 {
		t.Errorf("expected empty activeTextParts after finish-step, got %d", len(state.ActiveTextParts))
	}
	if len(state.ActiveReasoningParts) != 0 {
		t.Errorf("expected empty activeReasoningParts after finish-step, got %d", len(state.ActiveReasoningParts))
	}
}

func TestProcessUIMessageStream_ReasoningParts(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-3"}},
		{Type: ChunkReasoningStart, Fields: map[string]any{"id": "r1"}},
		{Type: ChunkReasoningDelta, Fields: map[string]any{"id": "r1", "delta": "thinking..."}},
		{Type: ChunkReasoningEnd, Fields: map[string]any{"id": "r1"}},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "answer"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkFinish},
	}

	state := NewStreamingUIMessageState("msg-3", nil)
	out := ProcessUIMessageStream(chunkSliceToChan(chunks), state)
	drainChunks(out)

	if len(state.Message.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d: %v", len(state.Message.Parts), state.Message.Parts)
	}
	if state.Message.Parts[0]["type"] != "reasoning" {
		t.Errorf("expected reasoning part, got %v", state.Message.Parts[0]["type"])
	}
	if state.Message.Parts[0]["text"] != "thinking..." {
		t.Errorf("expected 'thinking...', got %q", state.Message.Parts[0]["text"])
	}
	if state.Message.Parts[1]["type"] != "text" {
		t.Errorf("expected text part, got %v", state.Message.Parts[1]["type"])
	}
}

func TestProcessUIMessageStream_SourceParts(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-4"}},
		{Type: ChunkSourceURL, Fields: map[string]any{
			"sourceId": "s1", "url": "https://example.com", "title": "Example",
		}},
		{Type: ChunkSourceDocument, Fields: map[string]any{
			"sourceId": "s2", "title": "Doc", "mediaType": "text/plain", "filename": "doc.txt",
		}},
		{Type: ChunkFinish},
	}

	state := NewStreamingUIMessageState("msg-4", nil)
	out := ProcessUIMessageStream(chunkSliceToChan(chunks), state)
	drainChunks(out)

	if len(state.Message.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(state.Message.Parts))
	}
	if state.Message.Parts[0]["type"] != "source-url" {
		t.Errorf("expected source-url, got %v", state.Message.Parts[0]["type"])
	}
	if state.Message.Parts[1]["type"] != "source-document" {
		t.Errorf("expected source-document, got %v", state.Message.Parts[1]["type"])
	}
	if state.Message.Parts[1]["filename"] != "doc.txt" {
		t.Errorf("expected filename doc.txt, got %v", state.Message.Parts[1]["filename"])
	}
}

func TestProcessUIMessageStream_MessageMetadata(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-5"}},
		{Type: ChunkMessageMetadata, Fields: map[string]any{
			"messageMetadata": map[string]any{"model": "gpt-4"},
		}},
		{Type: ChunkFinish},
	}

	state := NewStreamingUIMessageState("msg-5", nil)
	out := ProcessUIMessageStream(chunkSliceToChan(chunks), state)
	drainChunks(out)

	if state.Message.Metadata == nil {
		t.Fatal("expected metadata to be set")
	}
	var meta map[string]any
	if err := json.Unmarshal(state.Message.Metadata, &meta); err != nil {
		t.Fatalf("unmarshal metadata: %v", err)
	}
	if meta["model"] != "gpt-4" {
		t.Errorf("expected model=gpt-4, got %v", meta["model"])
	}
}

func TestProcessUIMessageStream_FilePart(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-6"}},
		{Type: ChunkFile, Fields: map[string]any{
			"url": "https://example.com/file.png", "mediaType": "image/png", "name": "file.png",
		}},
		{Type: ChunkFinish},
	}

	state := NewStreamingUIMessageState("msg-6", nil)
	out := ProcessUIMessageStream(chunkSliceToChan(chunks), state)
	drainChunks(out)

	if len(state.Message.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(state.Message.Parts))
	}
	if state.Message.Parts[0]["type"] != "file" {
		t.Errorf("expected file part, got %v", state.Message.Parts[0]["type"])
	}
	if state.Message.Parts[0]["url"] != "https://example.com/file.png" {
		t.Errorf("expected file url, got %v", state.Message.Parts[0]["url"])
	}
}

func TestProcessUIMessageStream_ContinuationFromAssistant(t *testing.T) {
	lastMsg := &StreamingUIMessage{
		ID:   "existing-msg",
		Role: "assistant",
		Parts: []UIMessagePart{
			{"type": "text", "text": "previous "},
		},
	}

	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "existing-msg"}},
		{Type: ChunkStartStep},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "continued"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkFinishStep},
		{Type: ChunkFinish},
	}

	state := NewStreamingUIMessageState("existing-msg", lastMsg)
	out := ProcessUIMessageStream(chunkSliceToChan(chunks), state)
	drainChunks(out)

	// Should have: previous text + step-start + new text
	if len(state.Message.Parts) != 3 {
		t.Fatalf("expected 3 parts, got %d: %v", len(state.Message.Parts), state.Message.Parts)
	}
	if state.Message.Parts[0]["text"] != "previous " {
		t.Errorf("expected preserved previous text, got %v", state.Message.Parts[0]["text"])
	}
	if state.Message.Parts[1]["type"] != "step-start" {
		t.Errorf("expected step-start, got %v", state.Message.Parts[1]["type"])
	}
	if state.Message.Parts[2]["text"] != "continued" {
		t.Errorf("expected 'continued', got %v", state.Message.Parts[2]["text"])
	}
}
