package uistream

import (
	"sync"
	"testing"
)

func TestHandleUIMessageStreamFinish_OnFinishCalled(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-1"}},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "Hello World"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkFinish, Fields: map[string]any{"finishReason": "stop"}},
	}

	var finishInfo FinishInfo
	var mu sync.Mutex

	out := HandleUIMessageStreamFinish(chunkSliceToChan(chunks), HandleUIMessageStreamFinishOptions{
		MessageID: "msg-1",
		OnFinish: func(info FinishInfo) {
			mu.Lock()
			finishInfo = info
			mu.Unlock()
		},
	})
	drainChunks(out)

	mu.Lock()
	defer mu.Unlock()

	if finishInfo.IsAborted {
		t.Error("expected IsAborted=false")
	}
	if finishInfo.IsContinuation {
		t.Error("expected IsContinuation=false")
	}
	if finishInfo.ResponseMessage.ID != "msg-1" {
		t.Errorf("expected message ID msg-1, got %q", finishInfo.ResponseMessage.ID)
	}
	if finishInfo.FinishReason != "stop" {
		t.Errorf("expected finishReason=stop, got %q", finishInfo.FinishReason)
	}
	// Should have text part.
	if len(finishInfo.ResponseMessage.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(finishInfo.ResponseMessage.Parts))
	}
	if finishInfo.ResponseMessage.Parts[0]["text"] != "Hello World" {
		t.Errorf("expected 'Hello World', got %v", finishInfo.ResponseMessage.Parts[0]["text"])
	}
}

func TestHandleUIMessageStreamFinish_OnStepFinishCalled(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-2"}},
		{Type: ChunkStartStep},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "Step 1"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkFinishStep},
		{Type: ChunkStartStep},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t2"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t2", "delta": "Step 2"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t2"}},
		{Type: ChunkFinishStep},
		{Type: ChunkFinish, Fields: map[string]any{"finishReason": "stop"}},
	}

	var stepInfos []StepFinishInfo
	var mu sync.Mutex

	out := HandleUIMessageStreamFinish(chunkSliceToChan(chunks), HandleUIMessageStreamFinishOptions{
		MessageID: "msg-2",
		OnStepFinish: func(info StepFinishInfo) {
			mu.Lock()
			stepInfos = append(stepInfos, info)
			mu.Unlock()
		},
	})
	drainChunks(out)

	mu.Lock()
	defer mu.Unlock()

	if len(stepInfos) != 2 {
		t.Fatalf("expected 2 step finish callbacks, got %d", len(stepInfos))
	}

	// First step should have step-start + text part.
	step1Parts := stepInfos[0].ResponseMessage.Parts
	if len(step1Parts) != 2 { // step-start, text
		t.Fatalf("step 1: expected 2 parts, got %d: %v", len(step1Parts), step1Parts)
	}
	if step1Parts[0]["type"] != "step-start" {
		t.Errorf("step 1: expected step-start, got %v", step1Parts[0]["type"])
	}
	if step1Parts[1]["text"] != "Step 1" {
		t.Errorf("step 1: expected 'Step 1', got %v", step1Parts[1]["text"])
	}

	// Second step should have step-start*2 + text*2 (accumulated).
	step2Parts := stepInfos[1].ResponseMessage.Parts
	if len(step2Parts) != 4 { // step-start, text, step-start, text
		t.Fatalf("step 2: expected 4 parts, got %d: %v", len(step2Parts), step2Parts)
	}
}

func TestHandleUIMessageStreamFinish_AbortTracking(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-3"}},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "partial"}},
		{Type: ChunkAbort},
		{Type: ChunkFinish},
	}

	var finishInfo FinishInfo
	var mu sync.Mutex

	out := HandleUIMessageStreamFinish(chunkSliceToChan(chunks), HandleUIMessageStreamFinishOptions{
		MessageID: "msg-3",
		OnFinish: func(info FinishInfo) {
			mu.Lock()
			finishInfo = info
			mu.Unlock()
		},
	})
	drainChunks(out)

	mu.Lock()
	defer mu.Unlock()

	if !finishInfo.IsAborted {
		t.Error("expected IsAborted=true")
	}
}

func TestHandleUIMessageStreamFinish_Continuation(t *testing.T) {
	lastMsg := &StreamingUIMessage{
		ID:   "existing-assistant",
		Role: "assistant",
		Parts: []UIMessagePart{
			{"type": "text", "text": "previous "},
		},
	}

	chunks := []Chunk{
		{Type: ChunkStart}, // no messageId — should be injected
		{Type: ChunkStartStep},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "continued"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkFinishStep},
		{Type: ChunkFinish},
	}

	var finishInfo FinishInfo
	var mu sync.Mutex

	out := HandleUIMessageStreamFinish(chunkSliceToChan(chunks), HandleUIMessageStreamFinishOptions{
		MessageID:            "ignored-id",
		LastAssistantMessage: lastMsg,
		OnFinish: func(info FinishInfo) {
			mu.Lock()
			finishInfo = info
			mu.Unlock()
		},
	})

	// Verify messageId injection on start chunk.
	result := drainChunks(out)
	startChunk := result[0]
	if id, _ := startChunk.Fields["messageId"].(string); id != "existing-assistant" {
		t.Errorf("expected injected messageId=existing-assistant, got %q", id)
	}

	mu.Lock()
	defer mu.Unlock()

	if !finishInfo.IsContinuation {
		t.Error("expected IsContinuation=true")
	}
	if finishInfo.ResponseMessage.ID != "existing-assistant" {
		t.Errorf("expected message ID existing-assistant, got %q", finishInfo.ResponseMessage.ID)
	}
}

func TestHandleUIMessageStreamFinish_MessageIdInjection(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart}, // no messageId
		{Type: ChunkFinish},
	}

	out := HandleUIMessageStreamFinish(chunkSliceToChan(chunks), HandleUIMessageStreamFinishOptions{
		MessageID: "injected-123",
	})
	result := drainChunks(out)

	if id, _ := result[0].Fields["messageId"].(string); id != "injected-123" {
		t.Errorf("expected injected messageId=injected-123, got %q", id)
	}
}

func TestHandleUIMessageStreamFinish_ExistingMessageIdPreserved(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "original-id"}},
		{Type: ChunkFinish},
	}

	out := HandleUIMessageStreamFinish(chunkSliceToChan(chunks), HandleUIMessageStreamFinishOptions{
		MessageID: "should-not-override",
	})
	result := drainChunks(out)

	if id, _ := result[0].Fields["messageId"].(string); id != "original-id" {
		t.Errorf("expected preserved messageId=original-id, got %q", id)
	}
}

func TestHandleUIMessageStreamFinish_PassthroughWithoutCallbacks(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-pass"}},
		{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "text"}},
		{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}},
		{Type: ChunkFinish},
	}

	out := HandleUIMessageStreamFinish(chunkSliceToChan(chunks), HandleUIMessageStreamFinishOptions{
		MessageID: "msg-pass",
		// No callbacks.
	})
	result := drainChunks(out)

	if len(result) != len(chunks) {
		t.Errorf("expected %d chunks passthrough, got %d", len(chunks), len(result))
	}
}

func TestHandleUIMessageStreamFinish_OnFinishCalledOnce(t *testing.T) {
	chunks := []Chunk{
		{Type: ChunkStart, Fields: map[string]any{"messageId": "msg-once"}},
		{Type: ChunkFinish},
	}

	finishCount := 0
	var mu sync.Mutex

	out := HandleUIMessageStreamFinish(chunkSliceToChan(chunks), HandleUIMessageStreamFinishOptions{
		MessageID: "msg-once",
		OnFinish: func(info FinishInfo) {
			mu.Lock()
			finishCount++
			mu.Unlock()
		},
	})
	drainChunks(out)

	mu.Lock()
	defer mu.Unlock()

	if finishCount != 1 {
		t.Errorf("expected onFinish called once, got %d", finishCount)
	}
}
