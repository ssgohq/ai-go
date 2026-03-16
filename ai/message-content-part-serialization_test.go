package ai

import (
	"encoding/json"
	"testing"
)

// --- ContentPart constructors ---

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
	if p.FileURL != "https://example.com/doc.pdf" {
		t.Errorf("unexpected FileURL: %s", p.FileURL)
	}
	if p.MimeType != "application/pdf" {
		t.Errorf("unexpected MimeType: %s", p.MimeType)
	}
}

func TestReasoningPart(t *testing.T) {
	p := ReasoningPart("step 1: think about it")
	if p.Type != ContentPartTypeReasoning {
		t.Errorf("expected reasoning type, got %s", p.Type)
	}
	if p.ReasoningText != "step 1: think about it" {
		t.Errorf("unexpected ReasoningText: %s", p.ReasoningText)
	}
	// Text field must remain zero to avoid confusion.
	if p.Text != "" {
		t.Errorf("expected Text to be empty for reasoning part, got %q", p.Text)
	}
}

func TestToolCallPart(t *testing.T) {
	args := json.RawMessage(`{"q":"go"}`)
	p := ToolCallPart("tc-1", "search", args)
	if p.Type != ContentPartTypeToolCall {
		t.Errorf("expected tool_call type, got %s", p.Type)
	}
	if p.ToolCallID != "tc-1" {
		t.Errorf("unexpected ToolCallID: %s", p.ToolCallID)
	}
	if p.ToolCallName != "search" {
		t.Errorf("unexpected ToolCallName: %s", p.ToolCallName)
	}
	if string(p.ToolCallArgs) != `{"q":"go"}` {
		t.Errorf("unexpected ToolCallArgs: %s", p.ToolCallArgs)
	}
}

func TestToolResultPart(t *testing.T) {
	p := ToolResultPart("tc-1", "search", `{"results":[]}`)
	if p.Type != ContentPartTypeToolResult {
		t.Errorf("expected tool_result type, got %s", p.Type)
	}
	if p.ToolResultID != "tc-1" {
		t.Errorf("unexpected ToolResultID: %s", p.ToolResultID)
	}
	if p.ToolResultName != "search" {
		t.Errorf("unexpected ToolResultName: %s", p.ToolResultName)
	}
	if p.ToolResultOutput != `{"results":[]}` {
		t.Errorf("unexpected ToolResultOutput: %s", p.ToolResultOutput)
	}
}

func TestImageDataPart(t *testing.T) {
	data := []byte{0x89, 0x50, 0x4e, 0x47}
	p := ImageDataPart(data, "image/png")
	if p.Type != ContentPartTypeImage {
		t.Errorf("expected image_url type, got %s", p.Type)
	}
	if string(p.Data) != string(data) {
		t.Errorf("unexpected Data: %v", p.Data)
	}
	if p.MimeType != "image/png" {
		t.Errorf("unexpected MimeType: %s", p.MimeType)
	}
}

func TestImageFileIDPart(t *testing.T) {
	p := ImageFileIDPart("file-abc123")
	if p.Type != ContentPartTypeImage {
		t.Errorf("expected image_url type, got %s", p.Type)
	}
	if p.FileID != "file-abc123" {
		t.Errorf("unexpected FileID: %s", p.FileID)
	}
}

func TestFileDataPart(t *testing.T) {
	data := []byte("%PDF-1.4")
	p := FileDataPart(data, "application/pdf", "report.pdf")
	if p.Type != ContentPartTypeFile {
		t.Errorf("expected file type, got %s", p.Type)
	}
	if string(p.Data) != string(data) {
		t.Errorf("unexpected Data: %v", p.Data)
	}
	if p.MimeType != "application/pdf" {
		t.Errorf("unexpected MimeType: %s", p.MimeType)
	}
	if p.Filename != "report.pdf" {
		t.Errorf("unexpected Filename: %s", p.Filename)
	}
}

func TestFileIDPart(t *testing.T) {
	p := FileIDPart("file-xyz", "application/pdf")
	if p.Type != ContentPartTypeFile {
		t.Errorf("expected file type, got %s", p.Type)
	}
	if p.FileID != "file-xyz" {
		t.Errorf("unexpected FileID: %s", p.FileID)
	}
	if p.MimeType != "application/pdf" {
		t.Errorf("unexpected MimeType: %s", p.MimeType)
	}
}

func TestContentPartTypeImageAlias(t *testing.T) {
	if ContentPartTypeImage != ContentPartTypeImageURL {
		t.Errorf("ContentPartTypeImage should equal ContentPartTypeImageURL")
	}
}

// --- Round-trip tests for new multimodal constructors ---

func TestContentPartRoundTrip_ImageData(t *testing.T) {
	data := []byte{0x89, 0x50, 0x4e, 0x47}
	parts := []ContentPart{ImageDataPart(data, "image/png")}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeImage {
		t.Errorf("expected image_url type, got %s", rt[0].Type)
	}
	if string(rt[0].Data) != string(data) {
		t.Errorf("unexpected Data after round-trip: %v", rt[0].Data)
	}
	if rt[0].MimeType != "image/png" {
		t.Errorf("unexpected MimeType: %s", rt[0].MimeType)
	}
}

func TestContentPartRoundTrip_ImageFileID(t *testing.T) {
	parts := []ContentPart{ImageFileIDPart("file-abc123")}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeImage {
		t.Errorf("expected image_url type, got %s", rt[0].Type)
	}
	if rt[0].FileID != "file-abc123" {
		t.Errorf("unexpected FileID: %s", rt[0].FileID)
	}
}

func TestContentPartRoundTrip_FileData(t *testing.T) {
	data := []byte("%PDF-1.4")
	parts := []ContentPart{FileDataPart(data, "application/pdf", "report.pdf")}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeFile {
		t.Errorf("expected file type, got %s", rt[0].Type)
	}
	if string(rt[0].Data) != string(data) {
		t.Errorf("unexpected Data after round-trip: %v", rt[0].Data)
	}
	if rt[0].MimeType != "application/pdf" {
		t.Errorf("unexpected MimeType: %s", rt[0].MimeType)
	}
	if rt[0].Filename != "report.pdf" {
		t.Errorf("unexpected Filename: %s", rt[0].Filename)
	}
}

func TestContentPartRoundTrip_FileID(t *testing.T) {
	parts := []ContentPart{FileIDPart("file-xyz", "application/pdf")}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeFile {
		t.Errorf("expected file type, got %s", rt[0].Type)
	}
	if rt[0].FileID != "file-xyz" {
		t.Errorf("unexpected FileID: %s", rt[0].FileID)
	}
	if rt[0].MimeType != "application/pdf" {
		t.Errorf("unexpected MimeType: %s", rt[0].MimeType)
	}
}

// --- Message constructors ---

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

// --- ContentPart round-trip through engine adapter (via internal helpers) ---
// These tests verify that all part types survive the ai→engine→ai translation
// without data loss.

func TestContentPartRoundTrip_Text(t *testing.T) {
	parts := []ContentPart{TextPart("hello world")}
	roundTripped := fromEngineContentParts(toEngineContentParts(parts))
	if roundTripped[0].Type != ContentPartTypeText || roundTripped[0].Text != "hello world" {
		t.Errorf("text part round-trip failed: %+v", roundTripped[0])
	}
}

func TestContentPartRoundTrip_ImageURL(t *testing.T) {
	parts := []ContentPart{ImageURLPart("data:image/png;base64,abc")}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeImageURL || rt[0].ImageURL != "data:image/png;base64,abc" {
		t.Errorf("imageURL part round-trip failed: %+v", rt[0])
	}
}

func TestContentPartRoundTrip_File(t *testing.T) {
	parts := []ContentPart{FilePart("https://cdn.example.com/doc.pdf", "application/pdf")}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeFile {
		t.Errorf("expected file type, got %s", rt[0].Type)
	}
	if rt[0].FileURL != "https://cdn.example.com/doc.pdf" {
		t.Errorf("unexpected FileURL: %s", rt[0].FileURL)
	}
	if rt[0].MimeType != "application/pdf" {
		t.Errorf("unexpected MimeType: %s", rt[0].MimeType)
	}
}

func TestContentPartRoundTrip_Reasoning(t *testing.T) {
	parts := []ContentPart{ReasoningPart("I should search the web first.")}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeReasoning {
		t.Errorf("expected reasoning type, got %s", rt[0].Type)
	}
	if rt[0].ReasoningText != "I should search the web first." {
		t.Errorf("unexpected ReasoningText: %s", rt[0].ReasoningText)
	}
	if rt[0].Text != "" {
		t.Errorf("Text should be empty for reasoning part after round-trip, got %q", rt[0].Text)
	}
}

func TestContentPartRoundTrip_ToolCall(t *testing.T) {
	args := json.RawMessage(`{"query":"go lang"}`)
	parts := []ContentPart{ToolCallPart("call-abc", "web_search", args)}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeToolCall {
		t.Errorf("expected tool_call type, got %s", rt[0].Type)
	}
	if rt[0].ToolCallID != "call-abc" {
		t.Errorf("unexpected ToolCallID: %s", rt[0].ToolCallID)
	}
	if rt[0].ToolCallName != "web_search" {
		t.Errorf("unexpected ToolCallName: %s", rt[0].ToolCallName)
	}
	if string(rt[0].ToolCallArgs) != `{"query":"go lang"}` {
		t.Errorf("unexpected ToolCallArgs: %s", rt[0].ToolCallArgs)
	}
}

func TestContentPartRoundTrip_ToolResult(t *testing.T) {
	parts := []ContentPart{ToolResultPart("call-abc", "web_search", `{"results":["r1"]}`)}
	rt := fromEngineContentParts(toEngineContentParts(parts))
	if rt[0].Type != ContentPartTypeToolResult {
		t.Errorf("expected tool_result type, got %s", rt[0].Type)
	}
	if rt[0].ToolResultID != "call-abc" {
		t.Errorf("unexpected ToolResultID: %s", rt[0].ToolResultID)
	}
	if rt[0].ToolResultName != "web_search" {
		t.Errorf("unexpected ToolResultName: %s", rt[0].ToolResultName)
	}
	if rt[0].ToolResultOutput != `{"results":["r1"]}` {
		t.Errorf("unexpected ToolResultOutput: %s", rt[0].ToolResultOutput)
	}
}
