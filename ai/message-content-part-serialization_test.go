package ai

import "testing"

func TestTextPart(t *testing.T) {
	p := TextPart("hello")
	if p.Type != ContentPartTypeText {
		t.Errorf("expected text type, got %s", p.Type)
	}
	if p.Text != "hello" {
		t.Errorf("expected 'hello', got %s", p.Text)
	}
}

func TestImageURLPart(t *testing.T) {
	p := ImageURLPart("https://example.com/img.png")
	if p.Type != ContentPartTypeImageURL {
		t.Errorf("expected image_url type, got %s", p.Type)
	}
	if p.ImageURL != "https://example.com/img.png" {
		t.Errorf("unexpected ImageURL: %s", p.ImageURL)
	}
}

func TestFilePart(t *testing.T) {
	p := FilePart("https://example.com/doc.pdf", "application/pdf")
	if p.Type != ContentPartTypeFile {
		t.Errorf("expected file type, got %s", p.Type)
	}
	if p.MimeType != "application/pdf" {
		t.Errorf("unexpected MimeType: %s", p.MimeType)
	}
}

func TestUserMessage(t *testing.T) {
	m := UserMessage("hi")
	if m.Role != RoleUser {
		t.Errorf("expected user role, got %s", m.Role)
	}
	if len(m.Content) != 1 || m.Content[0].Text != "hi" {
		t.Error("expected single text part with 'hi'")
	}
}

func TestAssistantMessage(t *testing.T) {
	m := AssistantMessage("hello there")
	if m.Role != RoleAssistant {
		t.Errorf("expected assistant role, got %s", m.Role)
	}
}

func TestSystemMessage(t *testing.T) {
	m := SystemMessage("you are helpful")
	if m.Role != RoleSystem {
		t.Errorf("expected system role, got %s", m.Role)
	}
}
