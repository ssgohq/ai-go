package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestUploadFile_Success(t *testing.T) {
	want := UploadedFile{
		ID:       "file-abc123",
		Object:   "file",
		Filename: "report.pdf",
		Bytes:    1024,
		Status:   "uploaded",
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/files" {
			t.Errorf("expected /files, got %s", r.URL.Path)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
		}
		if r.FormValue("purpose") != string(FilePurposeUserData) {
			t.Errorf("expected purpose=user_data, got %q", r.FormValue("purpose"))
		}
		_, header, err := r.FormFile("file")
		if err != nil {
			t.Errorf("form file: %v", err)
		}
		if header.Filename != "report.pdf" {
			t.Errorf("expected filename=report.pdf, got %q", header.Filename)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(fileResponse{
			ID:       want.ID,
			Object:   want.Object,
			Filename: want.Filename,
			Bytes:    want.Bytes,
			Status:   want.Status,
		})
	}))
	defer srv.Close()

	model := NewLanguageModel("gpt-4o", Config{APIKey: "test-key", BaseURL: srv.URL})
	got, err := model.UploadFile(context.Background(), UploadFileRequest{
		Filename: "report.pdf",
		Purpose:  FilePurposeUserData,
		Data:     []byte("fake pdf content"),
		MimeType: "application/pdf",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID != want.ID {
		t.Errorf("expected ID=%q, got %q", want.ID, got.ID)
	}
	if got.Object != want.Object {
		t.Errorf("expected Object=%q, got %q", want.Object, got.Object)
	}
	if got.Filename != want.Filename {
		t.Errorf("expected Filename=%q, got %q", want.Filename, got.Filename)
	}
	if got.Bytes != want.Bytes {
		t.Errorf("expected Bytes=%d, got %d", want.Bytes, got.Bytes)
	}
	if got.Status != want.Status {
		t.Errorf("expected Status=%q, got %q", want.Status, got.Status)
	}
}

func TestUploadFile_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":{"message":"invalid api key","type":"authentication_error"}}`))
	}))
	defer srv.Close()

	model := NewLanguageModel("gpt-4o", Config{APIKey: "bad-key", BaseURL: srv.URL})
	_, err := model.UploadFile(context.Background(), UploadFileRequest{
		Filename: "file.txt",
		Purpose:  FilePurposeUserData,
		Data:     []byte("data"),
	})
	if err == nil {
		t.Fatal("expected error for HTTP 401, got nil")
	}
}

func TestUploadFile_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(fileResponse{
			Error: &struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			}{
				Code:    "invalid_request",
				Message: "unsupported file type",
			},
		})
	}))
	defer srv.Close()

	model := NewLanguageModel("gpt-4o", Config{APIKey: "test-key", BaseURL: srv.URL})
	_, err := model.UploadFile(context.Background(), UploadFileRequest{
		Filename: "file.exe",
		Purpose:  FilePurposeUserData,
		Data:     []byte("data"),
	})
	if err == nil {
		t.Fatal("expected error from API error body, got nil")
	}
}

func TestUploadFile_ValidationErrors(t *testing.T) {
	model := NewLanguageModel("gpt-4o", Config{APIKey: "test-key"})

	cases := []struct {
		name string
		req  UploadFileRequest
	}{
		{"missing filename", UploadFileRequest{Purpose: FilePurposeUserData, Data: []byte("x")}},
		{"missing purpose", UploadFileRequest{Filename: "f.txt", Data: []byte("x")}},
		{"empty data", UploadFileRequest{Filename: "f.txt", Purpose: FilePurposeUserData}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := model.UploadFile(context.Background(), tc.req)
			if err == nil {
				t.Errorf("expected validation error for %q, got nil", tc.name)
			}
		})
	}
}

func TestUploadFile_DefaultMimeType(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
		}
		_, header, err := r.FormFile("file")
		if err != nil {
			t.Errorf("form file: %v", err)
		}
		ct := header.Header.Get("Content-Type")
		if ct != "application/octet-stream" {
			t.Errorf("expected default mime type application/octet-stream, got %q", ct)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(fileResponse{
			ID: "file-def456", Object: "file", Filename: "data.bin", Bytes: 4, Status: "uploaded",
		})
	}))
	defer srv.Close()

	model := NewLanguageModel("gpt-4o", Config{APIKey: "test-key", BaseURL: srv.URL})
	_, err := model.UploadFile(context.Background(), UploadFileRequest{
		Filename: "data.bin",
		Purpose:  FilePurposeBatch,
		Data:     []byte("data"),
		// MimeType intentionally omitted to test default.
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
