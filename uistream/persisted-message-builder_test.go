package uistream

import (
	"encoding/json"
	"testing"
)

func TestPersistedMessageBuilder_TextPart(t *testing.T) {
	b := NewPersistedMessageBuilder()
	b.ObserveChunk(Chunk{Type: ChunkTextStart, Fields: map[string]any{"id": "t1"}})
	b.ObserveChunk(Chunk{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "Hello "}})
	b.ObserveChunk(Chunk{Type: ChunkTextDelta, Fields: map[string]any{"id": "t1", "delta": "world"}})
	b.ObserveChunk(Chunk{Type: ChunkTextEnd, Fields: map[string]any{"id": "t1"}})

	if got := b.Content(); got != "Hello world" {
		t.Errorf("Content() = %q, want %q", got, "Hello world")
	}

	var parts []map[string]any
	if err := json.Unmarshal(b.Parts(), &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 1 || parts[0]["type"] != "text" || parts[0]["text"] != "Hello world" {
		t.Errorf("unexpected parts: %v", parts)
	}
}

func TestPersistedMessageBuilder_ReasoningPart(t *testing.T) {
	b := NewPersistedMessageBuilder()
	b.ObserveChunk(Chunk{Type: ChunkReasoningStart, Fields: map[string]any{"id": "t1"}})
	b.ObserveChunk(Chunk{Type: ChunkReasoningDelta, Fields: map[string]any{"id": "t1", "delta": "I think"}})
	b.ObserveChunk(Chunk{Type: ChunkReasoningEnd, Fields: map[string]any{"id": "t1", "signature": "sig123"}})

	var parts []map[string]any
	if err := json.Unmarshal(b.Parts(), &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
	p := parts[0]
	if p["type"] != "reasoning" || p["reasoning"] != "I think" || p["signature"] != "sig123" {
		t.Errorf("unexpected reasoning part: %v", p)
	}
}

func TestPersistedMessageBuilder_ToolInvocation(t *testing.T) {
	b := NewPersistedMessageBuilder()
	input := map[string]any{"query": "golang"}
	output := []any{map[string]any{"doc": "1"}}

	b.ObserveChunk(Chunk{Type: ChunkToolInputAvailable, Fields: map[string]any{
		"toolCallId": "call_1",
		"toolName":   "search",
		"input":      input,
	}})
	b.ObserveChunk(Chunk{Type: ChunkToolOutputAvailable, Fields: map[string]any{
		"toolCallId": "call_1",
		"output":     output,
	}})

	var parts []map[string]any
	if err := json.Unmarshal(b.Parts(), &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
	p := parts[0]
	if p["type"] != "tool-invocation" || p["state"] != "output-available" {
		t.Errorf("unexpected tool part: %v", p)
	}
}

func TestPersistedMessageBuilder_SourceURL(t *testing.T) {
	b := NewPersistedMessageBuilder()
	b.ObserveChunk(Chunk{Type: ChunkSourceURL, Fields: map[string]any{
		"sourceId": "src_1",
		"url":      "https://example.com",
		"title":    "Example",
	}})

	var parts []map[string]any
	if err := json.Unmarshal(b.Parts(), &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 1 || parts[0]["type"] != "source-url" {
		t.Errorf("unexpected parts: %v", parts)
	}
	if parts[0]["id"] != "src_1" {
		t.Errorf("expected id=src_1, got %v", parts[0]["id"])
	}
}

func TestPersistedMessageBuilder_DataPart(t *testing.T) {
	b := NewPersistedMessageBuilder()
	b.ObserveChunk(Chunk{Type: "data-research_plan", Fields: map[string]any{
		"data": map[string]any{"steps": []string{"a", "b"}},
	}})
	// transient data should be excluded
	b.ObserveChunk(Chunk{Type: "data-transient_thing", Fields: map[string]any{
		"data":      "ignored",
		"transient": true,
	}})

	var parts []map[string]any
	if err := json.Unmarshal(b.Parts(), &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 1 {
		t.Fatalf("expected 1 part (transient excluded), got %d", len(parts))
	}
	p := parts[0]
	if p["type"] != "data" || p["name"] != "research_plan" || p["isTransient"] != false {
		t.Errorf("unexpected data part: %v", p)
	}
}

func TestPersistedMessageBuilder_TransientDataExcluded(t *testing.T) {
	b := NewPersistedMessageBuilder()
	b.ObserveChunk(Chunk{Type: "transient-data-activity", Fields: map[string]any{
		"data": "should be excluded",
	}})

	if p := b.Parts(); p != nil {
		t.Errorf("expected nil parts for transient-only data, got %s", p)
	}
}

func TestPersistedMessageBuilder_MessageMetadata(t *testing.T) {
	b := NewPersistedMessageBuilder()
	b.ObserveChunk(Chunk{Type: ChunkMessageMetadata, Fields: map[string]any{
		"messageMetadata": map[string]any{"model": "gpt-4o"},
	}})

	meta := b.Metadata()
	if meta == nil {
		t.Fatal("expected metadata, got nil")
	}
	var m map[string]any
	if err := json.Unmarshal(meta, &m); err != nil {
		t.Fatal(err)
	}
	if m["model"] != "gpt-4o" {
		t.Errorf("unexpected metadata: %v", m)
	}
}

func TestPersistedMessageBuilder_NilPartsWhenEmpty(t *testing.T) {
	b := NewPersistedMessageBuilder()
	if p := b.Parts(); p != nil {
		t.Errorf("expected nil parts for empty builder, got %s", p)
	}
	if c := b.Content(); c != "" {
		t.Errorf("expected empty content, got %q", c)
	}
}

func TestMergeWithPersistence_IntegrationOption(t *testing.T) {
	b := NewPersistedMessageBuilder()
	opt := MergeWithPersistence(b)

	cfg := &mergeConfig{}
	opt(cfg)

	if cfg.persistenceBuilder == nil {
		t.Error("persistenceBuilder should be set in mergeConfig")
	}
	if cfg.persistenceBuilder != b {
		t.Error("persistenceBuilder should be the same instance")
	}
}
