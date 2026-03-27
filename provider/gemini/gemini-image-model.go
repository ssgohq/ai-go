package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/open-ai-sdk/ai-go/ai"
)

// ImageModel implements ai.ImageModel using the native Gemini API's
// non-streaming :generateContent endpoint. Unlike NativeLanguageModel,
// this uses the synchronous endpoint since images are returned as
// complete base64 blobs.
//
// Use NewImageModel to construct an instance.
type ImageModel struct {
	modelID string
	cfg     Config
	client  *http.Client
}

// NewImageModel creates a Gemini-backed ai.ImageModel that generates images
// using the native Gemini API.
//
// Supported models: gemini-2.5-flash-image, gemini-3-pro-image-preview,
// gemini-3.1-flash-image-preview.
func NewImageModel(modelID string, cfg Config) *ImageModel {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	return &ImageModel{
		modelID: modelID,
		cfg:     cfg,
		client:  &http.Client{Timeout: timeout},
	}
}

// ModelID returns the Gemini model identifier.
func (m *ImageModel) ModelID() string { return m.modelID }

// Generate sends a request to the Gemini API and returns generated images.
func (m *ImageModel) Generate(ctx context.Context, req ai.GenerateImageRequest) (*ai.GenerateImageResult, error) {
	nr := m.buildRequest(req)

	body, err := json.Marshal(nr)
	if err != nil {
		return nil, fmt.Errorf("gemini-image: marshal request: %w", err)
	}

	baseURL := m.cfg.BaseURL
	if baseURL == "" {
		baseURL = nativeBaseURL
	}
	url := fmt.Sprintf("%s/models/%s:generateContent", baseURL, m.modelID)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini-image: build http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", m.cfg.APIKey)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini-image: http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("gemini-image: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini-image: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	return m.parseResponse(respBody)
}

// imageGenerateResponse is the JSON response from the non-streaming generateContent endpoint.
type imageGenerateResponse struct {
	Candidates    []imageCandidate     `json:"candidates"`
	UsageMetadata *nativeUsageMetadata `json:"usageMetadata"`
}

type imageCandidate struct {
	Content *imageCandidateContent `json:"content"`
}

type imageCandidateContent struct {
	Parts []imageResponsePart `json:"parts"`
	Role  string              `json:"role"`
}

type imageResponsePart struct {
	Text       string            `json:"text,omitempty"`
	InlineData *nativeInlineData `json:"inlineData,omitempty"`
}

func (m *ImageModel) buildRequest(req ai.GenerateImageRequest) nativeRequest {
	nr := nativeRequest{}

	var parts []nativePart

	if req.Prompt != "" {
		parts = append(parts, nativePart{Text: req.Prompt})
	}

	for _, img := range req.Images {
		if len(img.Data) > 0 {
			parts = append(parts, nativePart{
				InlineData: &nativeInlineData{
					MimeType: img.MimeType,
					Data:     base64.StdEncoding.EncodeToString(img.Data),
				},
			})
		} else if img.URL != "" {
			parts = append(parts, encodeMediaFromURL(img.URL, img.MimeType))
		}
	}

	nr.Contents = []nativeContent{
		{Role: "user", Parts: parts},
	}

	genCfg := &nativeGenerationConfig{
		ResponseModalities: []string{"IMAGE"},
	}

	if req.AspectRatio != "" || req.Size != "" {
		ic := &nativeImageConfig{}
		if req.AspectRatio != "" {
			ic.AspectRatio = req.AspectRatio
		}
		if req.Size != "" {
			ic.ImageSize = req.Size
		}
		genCfg.ImageConfig = ic
	}

	if req.Seed != nil {
		genCfg.Seed = req.Seed
	}

	nr.GenerationConfig = genCfg

	return nr
}

func (m *ImageModel) parseResponse(data []byte) (*ai.GenerateImageResult, error) {
	var resp imageGenerateResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("gemini-image: unmarshal response: %w", err)
	}

	result := &ai.GenerateImageResult{}

	if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			if part.InlineData != nil {
				decoded, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
				if err != nil {
					continue
				}
				result.Images = append(result.Images, ai.GeneratedImage{
					Data:     decoded,
					MimeType: part.InlineData.MimeType,
				})
			}
		}
	}

	if resp.UsageMetadata != nil {
		result.Usage = &ai.Usage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		}
	}

	return result, nil
}
