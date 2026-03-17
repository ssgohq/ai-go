package uistream_test

import (
	"context"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
	"github.com/open-ai-sdk/ai-go/uistream"
)

// stubModel is a minimal ai.LanguageModel for envelope converter tests.
type stubModel struct{ id string }

func (s *stubModel) ModelID() string { return s.id }
func (s *stubModel) Stream(
	_ context.Context, _ ai.LanguageModelRequest,
) (<-chan ai.StreamEvent, error) {
	return nil, nil
}

func TestToGenerateTextRequest_BasicMessages(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		ID: "session-1",
		Messages: []uistream.EnvelopeMessage{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there"},
		},
	}

	model := &stubModel{id: "gpt-4o"}
	req := uistream.ToGenerateTextRequest(env, model)

	if req.Model != model {
		t.Error("expected model to be set on request")
	}
	if len(req.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(req.Messages))
	}
	if req.Messages[0].Role != ai.RoleUser {
		t.Errorf("first message role: got %q, want %q", req.Messages[0].Role, ai.RoleUser)
	}
	if req.Messages[1].Role != ai.RoleAssistant {
		t.Errorf("second message role: got %q, want %q", req.Messages[1].Role, ai.RoleAssistant)
	}
}

func TestToGenerateTextRequest_BodyHints_System(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{{Role: "user", Content: "Hi"}},
		Body:     map[string]any{"system": "You are a helpful assistant."},
	}

	req := uistream.ToGenerateTextRequest(env, &stubModel{id: "m"})

	if req.System != "You are a helpful assistant." {
		t.Errorf("System: got %q", req.System)
	}
}

func TestToGenerateTextRequest_BodyHints_MaxSteps(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{{Role: "user", Content: "Hi"}},
		Body:     map[string]any{"maxSteps": float64(5)},
	}

	req := uistream.ToGenerateTextRequest(env, &stubModel{id: "m"})

	if req.MaxSteps != 5 {
		t.Errorf("MaxSteps: got %d, want 5", req.MaxSteps)
	}
}

func TestToGenerateTextRequest_BodyHints_MaxTokens(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{{Role: "user", Content: "Hi"}},
		Body:     map[string]any{"maxTokens": float64(1024)},
	}

	req := uistream.ToGenerateTextRequest(env, &stubModel{id: "m"})

	if req.Settings.MaxTokens != 1024 {
		t.Errorf("Settings.MaxTokens: got %d, want 1024", req.Settings.MaxTokens)
	}
}

func TestToGenerateTextRequest_NilBody(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{{Role: "user", Content: "Hi"}},
	}

	// Should not panic.
	req := uistream.ToGenerateTextRequest(env, &stubModel{id: "m"})

	if req.System != "" {
		t.Errorf("expected empty system, got %q", req.System)
	}
}

func TestToGenerateTextRequestFromRegistry_ResolvesModel(t *testing.T) {
	registry := ai.NewRegistry()
	registry.Register("openai", func(modelID string) ai.LanguageModel {
		return &stubModel{id: "openai:" + modelID}
	})

	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{{Role: "user", Content: "Hi"}},
		Body:     map[string]any{"modelId": "openai:gpt-4o"},
	}

	req, err := uistream.ToGenerateTextRequestFromRegistry(env, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model.ModelID() != "openai:gpt-4o" {
		t.Errorf("Model.ModelID: got %q, want %q", req.Model.ModelID(), "openai:gpt-4o")
	}
}

func TestToGenerateTextRequestFromRegistry_MissingModelID(t *testing.T) {
	registry := ai.NewRegistry()

	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{{Role: "user", Content: "Hi"}},
		Body:     map[string]any{},
	}

	_, err := uistream.ToGenerateTextRequestFromRegistry(env, registry)
	if err == nil {
		t.Fatal("expected error when modelId missing, got nil")
	}
}

func TestToGenerateTextRequestFromRegistry_UnknownPrefix(t *testing.T) {
	registry := ai.NewRegistry()

	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{{Role: "user", Content: "Hi"}},
		Body:     map[string]any{"modelId": "unknown:model"},
	}

	_, err := uistream.ToGenerateTextRequestFromRegistry(env, registry)
	if err == nil {
		t.Fatal("expected error for unknown prefix, got nil")
	}
}

func TestChatRequestEnvelope_TriggerAndMessageID_RoundTrip(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		ID:        "sess-1",
		Trigger:   "regenerate-message",
		MessageID: "msg-regen-42",
		Messages: []uistream.EnvelopeMessage{
			{Role: "user", Content: "Hi", Metadata: map[string]any{"clientTime": "12:00"}},
		},
	}

	if env.Trigger != "regenerate-message" {
		t.Errorf("Trigger: got %q, want %q", env.Trigger, "regenerate-message")
	}
	if env.MessageID != "msg-regen-42" {
		t.Errorf("MessageID: got %q, want %q", env.MessageID, "msg-regen-42")
	}
	meta, ok := env.Messages[0].Metadata["clientTime"]
	if !ok || meta != "12:00" {
		t.Errorf("per-message metadata: got %v", env.Messages[0].Metadata)
	}
}

func TestResolveMessageIDFromEnvelope_PrefersEnvelopeMessageID(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		MessageID: "explicit-msg-id",
		Messages: []uistream.EnvelopeMessage{
			{Role: "assistant", ID: "last-assistant-id"},
		},
	}
	got := uistream.ResolveMessageIDFromEnvelope(env, "fallback-id")
	if got != "explicit-msg-id" {
		t.Errorf("expected explicit MessageID to win, got %q", got)
	}
}

func TestResolveMessageIDFromEnvelope_FallsBackToLastAssistant(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		// MessageID not set
		Messages: []uistream.EnvelopeMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", ID: "asst-msg-99"},
		},
	}
	got := uistream.ResolveMessageIDFromEnvelope(env, "fallback-id")
	if got != "asst-msg-99" {
		t.Errorf("expected last assistant ID, got %q", got)
	}
}

func TestResolveMessageIDFromEnvelope_FallbackWhenNeitherSet(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{
			{Role: "user", Content: "hi"},
		},
	}
	got := uistream.ResolveMessageIDFromEnvelope(env, "generated-id")
	if got != "generated-id" {
		t.Errorf("expected fallback, got %q", got)
	}
}

func TestToGenerateTextRequest_MessageParts(t *testing.T) {
	env := uistream.ChatRequestEnvelope{
		Messages: []uistream.EnvelopeMessage{
			{
				Role: "user",
				Parts: []uistream.EnvelopePartUnion{
					{Type: uistream.EnvelopePartTypeText, Text: "What's in this image?"},
					{Type: uistream.EnvelopePartTypeImage, URL: "https://example.com/img.png"},
				},
			},
		},
	}

	req := uistream.ToGenerateTextRequest(env, &stubModel{id: "m"})

	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(req.Messages))
	}
	if len(req.Messages[0].Content) != 2 {
		t.Errorf("expected 2 content parts, got %d", len(req.Messages[0].Content))
	}
}
