package ai_test

import (
	"context"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// mockModel is a minimal LanguageModel for registry tests.
type mockModel struct{ id string }

func (m *mockModel) ModelID() string { return m.id }
func (m *mockModel) Stream(_ context.Context, _ ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	return nil, nil
}

func newMockFactory(prefix string) func(modelID string) ai.LanguageModel {
	return func(modelID string) ai.LanguageModel {
		return &mockModel{id: prefix + ":" + modelID}
	}
}

func TestRegistry_ResolveWithPrefix(t *testing.T) {
	r := ai.NewRegistry()
	r.Register("openai", newMockFactory("openai"))
	r.Register("gemini", newMockFactory("gemini"))

	m, err := r.Model("openai:gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.ModelID() != "openai:gpt-4o" {
		t.Errorf("got ModelID %q, want %q", m.ModelID(), "openai:gpt-4o")
	}

	m2, err := r.Model("gemini:gemini-2.0-flash")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m2.ModelID() != "gemini:gemini-2.0-flash" {
		t.Errorf("got ModelID %q, want %q", m2.ModelID(), "gemini:gemini-2.0-flash")
	}
}

func TestRegistry_ResolveWithDefaultPrefix(t *testing.T) {
	r := ai.NewRegistry()
	r.Register("openai", newMockFactory("openai"))
	r.SetDefault("openai")

	m, err := r.Model("gpt-4o-mini")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.ModelID() != "openai:gpt-4o-mini" {
		t.Errorf("got ModelID %q, want %q", m.ModelID(), "openai:gpt-4o-mini")
	}
}

func TestRegistry_ErrorOnUnknownPrefix(t *testing.T) {
	r := ai.NewRegistry()

	_, err := r.Model("unknown:model")
	if err == nil {
		t.Fatal("expected error for unknown prefix, got nil")
	}
}

func TestRegistry_ErrorOnMissingDefaultPrefix(t *testing.T) {
	r := ai.NewRegistry()
	r.Register("openai", newMockFactory("openai"))

	_, err := r.Model("gpt-4o") // no prefix, no default set
	if err == nil {
		t.Fatal("expected error when no default prefix set, got nil")
	}
}

func TestRegistry_MultipleFactories(t *testing.T) {
	r := ai.NewRegistry()
	prefixes := []string{"openai", "anthropic", "gemini"}
	for _, p := range prefixes {
		p := p
		r.Register(p, newMockFactory(p))
	}

	for _, p := range prefixes {
		id := p + ":test-model"
		m, err := r.Model(id)
		if err != nil {
			t.Errorf("prefix %q: unexpected error: %v", p, err)
			continue
		}
		if m.ModelID() != id {
			t.Errorf("prefix %q: got %q, want %q", p, m.ModelID(), id)
		}
	}
}
