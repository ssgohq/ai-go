package uistream

import (
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// TestToAIContentParts_ImageFileID verifies that an image envelope part with a
// FileID produces an ImageFileIDPart (FileID priority).
func TestToAIContentParts_ImageFileID(t *testing.T) {
	parts := []EnvelopePartUnion{
		{Type: EnvelopePartTypeImage, FileID: "file-abc123", MediaType: "image/png"},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	p := got[0]
	if p.Type != ai.ContentPartTypeImage {
		t.Errorf("expected type image_url, got %q", p.Type)
	}
	if p.FileID != "file-abc123" {
		t.Errorf("expected FileID=file-abc123, got %q", p.FileID)
	}
	if p.ImageURL != "" {
		t.Errorf("expected ImageURL empty, got %q", p.ImageURL)
	}
	if len(p.Data) != 0 {
		t.Errorf("expected Data empty, got %v", p.Data)
	}
}

// TestToAIContentParts_ImageData verifies that an image envelope part with inline
// Data (and no FileID) produces an ImageDataPart.
func TestToAIContentParts_ImageData(t *testing.T) {
	data := []byte{0x89, 0x50, 0x4e, 0x47}
	parts := []EnvelopePartUnion{
		{Type: EnvelopePartTypeImage, Data: data, MediaType: "image/png"},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	p := got[0]
	if p.Type != ai.ContentPartTypeImage {
		t.Errorf("expected type image_url, got %q", p.Type)
	}
	if string(p.Data) != string(data) {
		t.Errorf("expected Data=%v, got %v", data, p.Data)
	}
	if p.MimeType != "image/png" {
		t.Errorf("expected MimeType=image/png, got %q", p.MimeType)
	}
	if p.FileID != "" {
		t.Errorf("expected FileID empty, got %q", p.FileID)
	}
}

// TestToAIContentParts_ImageURL verifies that an image envelope part with only a
// URL falls through to produce an ImageURLPart (existing behaviour).
func TestToAIContentParts_ImageURL(t *testing.T) {
	parts := []EnvelopePartUnion{
		{Type: EnvelopePartTypeImage, URL: "https://example.com/img.png", MediaType: "image/png"},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	p := got[0]
	if p.Type != ai.ContentPartTypeImage {
		t.Errorf("expected type image_url, got %q", p.Type)
	}
	if p.ImageURL != "https://example.com/img.png" {
		t.Errorf("expected ImageURL=https://example.com/img.png, got %q", p.ImageURL)
	}
}

// TestToAIContentParts_FileFileID verifies that a file envelope part with a
// FileID produces a FileIDPart (FileID priority).
func TestToAIContentParts_FileFileID(t *testing.T) {
	parts := []EnvelopePartUnion{
		{Type: EnvelopePartTypeFile, FileID: "file-xyz", MediaType: "application/pdf"},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	p := got[0]
	if p.Type != ai.ContentPartTypeFile {
		t.Errorf("expected type file, got %q", p.Type)
	}
	if p.FileID != "file-xyz" {
		t.Errorf("expected FileID=file-xyz, got %q", p.FileID)
	}
	if p.MimeType != "application/pdf" {
		t.Errorf("expected MimeType=application/pdf, got %q", p.MimeType)
	}
}

// TestToAIContentParts_FileData verifies that a file envelope part with inline
// Data produces a FileDataPart with the correct Filename.
func TestToAIContentParts_FileData(t *testing.T) {
	data := []byte("%PDF-1.4")
	parts := []EnvelopePartUnion{
		{Type: EnvelopePartTypeFile, Data: data, MediaType: "application/pdf", Name: "report.pdf"},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	p := got[0]
	if p.Type != ai.ContentPartTypeFile {
		t.Errorf("expected type file, got %q", p.Type)
	}
	if string(p.Data) != string(data) {
		t.Errorf("expected Data=%v, got %v", data, p.Data)
	}
	if p.MimeType != "application/pdf" {
		t.Errorf("expected MimeType=application/pdf, got %q", p.MimeType)
	}
	if p.Filename != "report.pdf" {
		t.Errorf("expected Filename=report.pdf, got %q", p.Filename)
	}
}

// TestToAIContentParts_FileURL verifies that a file envelope part with only a
// URL produces a FilePart with Filename set (existing behaviour preserved).
func TestToAIContentParts_FileURL(t *testing.T) {
	parts := []EnvelopePartUnion{
		{Type: EnvelopePartTypeFile, URL: "https://cdn.example.com/doc.pdf", MediaType: "application/pdf", Name: "doc.pdf"},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	p := got[0]
	if p.Type != ai.ContentPartTypeFile {
		t.Errorf("expected type file, got %q", p.Type)
	}
	if p.FileURL != "https://cdn.example.com/doc.pdf" {
		t.Errorf("expected FileURL=https://cdn.example.com/doc.pdf, got %q", p.FileURL)
	}
	if p.MimeType != "application/pdf" {
		t.Errorf("expected MimeType=application/pdf, got %q", p.MimeType)
	}
	if p.Filename != "doc.pdf" {
		t.Errorf("expected Filename=doc.pdf, got %q", p.Filename)
	}
}

// TestToAIContentParts_TextPart verifies plain text parts are passed through.
func TestToAIContentParts_TextPart(t *testing.T) {
	parts := []EnvelopePartUnion{
		{Type: EnvelopePartTypeText, Text: "hello world"},
	}
	got := ToAIContentParts(parts)
	if len(got) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got))
	}
	p := got[0]
	if p.Type != ai.ContentPartTypeText {
		t.Errorf("expected type text, got %q", p.Type)
	}
	if p.Text != "hello world" {
		t.Errorf("expected Text=hello world, got %q", p.Text)
	}
}

// TestToAIContentParts_FileID_Priority verifies FileID wins over Data when both
// are set on an image part.
func TestToAIContentParts_FileID_Priority(t *testing.T) {
	parts := []EnvelopePartUnion{
		{
			Type:      EnvelopePartTypeImage,
			FileID:    "file-priority",
			Data:      []byte{0x01, 0x02},
			MediaType: "image/png",
		},
	}
	got := ToAIContentParts(parts)
	if got[0].FileID != "file-priority" {
		t.Errorf("expected FileID to win over Data, got FileID=%q Data=%v", got[0].FileID, got[0].Data)
	}
	if len(got[0].Data) != 0 {
		t.Errorf("expected Data empty when FileID set, got %v", got[0].Data)
	}
}

// TestToAIMessages_TextContent verifies the Content shorthand path for simple
// string-only messages.
func TestToAIMessages_TextContent(t *testing.T) {
	msgs := []EnvelopeMessage{
		{Role: "user", Content: "hello"},
	}
	got := ToAIMessages(msgs)
	if len(got) != 1 {
		t.Fatalf("expected 1 message, got %d", len(got))
	}
	if got[0].Role != ai.RoleUser {
		t.Errorf("expected role user, got %q", got[0].Role)
	}
	if len(got[0].Content) != 1 || got[0].Content[0].Text != "hello" {
		t.Errorf("expected single text part 'hello', got %+v", got[0].Content)
	}
}

// TestToAIMessages_PartsOverrideContent verifies that when Parts is non-empty,
// Content is ignored.
func TestToAIMessages_PartsOverrideContent(t *testing.T) {
	msgs := []EnvelopeMessage{
		{
			Role:    "user",
			Content: "should be ignored",
			Parts: []EnvelopePartUnion{
				{Type: EnvelopePartTypeText, Text: "from parts"},
			},
		},
	}
	got := ToAIMessages(msgs)
	if len(got[0].Content) != 1 || got[0].Content[0].Text != "from parts" {
		t.Errorf("expected parts to override content, got %+v", got[0].Content)
	}
}
