package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"time"
)

// UploadedFile holds the metadata returned by the OpenAI /v1/files endpoint
// after a successful upload.
type UploadedFile struct {
	// ID is the OpenAI file identifier (e.g. "file-abc123").
	ID string
	// Object is always "file" for successfully uploaded files.
	Object string
	// Filename is the name provided during upload.
	Filename string
	// Bytes is the size of the uploaded file in bytes.
	Bytes int
	// Status is the processing status: "uploaded", "processed", or "error".
	Status string
}

// FilePurpose identifies the intended use of an uploaded file.
type FilePurpose string

const (
	// FilePurposeUserData is used for files passed to model inputs via file_id.
	FilePurposeUserData FilePurpose = "user_data"
	// FilePurposeAssistants is used for files associated with Assistants.
	FilePurposeAssistants FilePurpose = "assistants"
	// FilePurposeFineTune is used for fine-tuning dataset files.
	FilePurposeFineTune FilePurpose = "fine-tune"
	// FilePurposeBatch is used for Batch API input files.
	FilePurposeBatch FilePurpose = "batch"
	// FilePurposeVision is used for vision fine-tuning.
	FilePurposeVision FilePurpose = "vision"
)

// UploadFileRequest holds the parameters for a file upload.
type UploadFileRequest struct {
	// Filename is the name to assign the file on the OpenAI platform.
	Filename string
	// Purpose identifies the intended use of the file.
	Purpose FilePurpose
	// Data is the raw file content to upload.
	Data []byte
	// MimeType is the MIME type of the file (e.g. "application/pdf").
	// Defaults to "application/octet-stream" when empty.
	MimeType string
}

// fileClient performs file upload operations against the OpenAI Files API.
type fileClient struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// newFileClient creates a file client from a LanguageModel's config.
func newFileClient(apiKey, baseURL string) *fileClient {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &fileClient{
		apiKey:  apiKey,
		baseURL: baseURL,
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

// UploadFile uploads data to the OpenAI /v1/files endpoint and returns the
// resulting file metadata. It supports all FilePurpose values including
// user_data for multimodal model inputs via file_id.
func (m *LanguageModel) UploadFile(ctx context.Context, req UploadFileRequest) (*UploadedFile, error) {
	fc := newFileClient(m.apiKey, m.baseURL)
	return fc.upload(ctx, req)
}

func (fc *fileClient) upload(ctx context.Context, req UploadFileRequest) (*UploadedFile, error) {
	if req.Filename == "" {
		return nil, fmt.Errorf("openai: upload file: filename is required")
	}
	if req.Purpose == "" {
		return nil, fmt.Errorf("openai: upload file: purpose is required")
	}
	if len(req.Data) == 0 {
		return nil, fmt.Errorf("openai: upload file: data is empty")
	}

	mimeType := req.MimeType
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}

	body, contentType, err := buildMultipartBody(req.Filename, req.Purpose, req.Data, mimeType)
	if err != nil {
		return nil, fmt.Errorf("openai: upload file: build multipart: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fc.baseURL+"/files", body)
	if err != nil {
		return nil, fmt.Errorf("openai: upload file: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", contentType)
	httpReq.Header.Set("Authorization", "Bearer "+fc.apiKey)

	resp, err := fc.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: upload file: http: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: upload file: read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("openai: upload file: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	return parseFileResponse(respBody)
}

// buildMultipartBody constructs the multipart/form-data body for the /v1/files endpoint.
func buildMultipartBody(
	filename string, purpose FilePurpose, data []byte, mimeType string,
) (*bytes.Buffer, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Add "purpose" field.
	if err := w.WriteField("purpose", string(purpose)); err != nil {
		return nil, "", err
	}

	// Add "file" part with explicit Content-Type header.
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename=%q`, filename))
	h.Set("Content-Type", mimeType)
	part, err := w.CreatePart(h)
	if err != nil {
		return nil, "", err
	}
	if _, err = part.Write(data); err != nil {
		return nil, "", err
	}

	if err = w.Close(); err != nil {
		return nil, "", err
	}
	return &buf, w.FormDataContentType(), nil
}

// fileResponse mirrors the JSON object returned by the OpenAI Files API.
type fileResponse struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	Filename string `json:"filename"`
	Bytes    int    `json:"bytes"`
	Status   string `json:"status"`
	Error    *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func parseFileResponse(body []byte) (*UploadedFile, error) {
	var r fileResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("openai: upload file: decode response: %w", err)
	}
	if r.Error != nil {
		return nil, fmt.Errorf("openai: upload file: %s: %s", r.Error.Code, r.Error.Message)
	}
	return &UploadedFile{
		ID:       r.ID,
		Object:   r.Object,
		Filename: r.Filename,
		Bytes:    r.Bytes,
		Status:   r.Status,
	}, nil
}
