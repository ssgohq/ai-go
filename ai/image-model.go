package ai

import (
	"context"
	"encoding/base64"
)

// ImageModel is the interface a provider must implement for image generation.
// Unlike LanguageModel which is stream-based, ImageModel uses a synchronous
// Generate method since images are returned as complete blobs.
type ImageModel interface {
	// ModelID returns the provider-specific model identifier.
	ModelID() string

	// Generate creates images based on the given request.
	Generate(ctx context.Context, req GenerateImageRequest) (*GenerateImageResult, error)
}

// GenerateImageRequest is the input to ImageModel.Generate and GenerateImage.
type GenerateImageRequest struct {
	// Model is the image model to call.
	Model ImageModel
	// Prompt is the text description of the image to generate.
	Prompt string
	// N is the number of images to generate (default 1).
	N int
	// AspectRatio is the desired aspect ratio (e.g. "1:1", "16:9", "3:4").
	AspectRatio string
	// Size is the desired image size (e.g. "1K", "2K").
	Size string
	// Seed is an optional seed for reproducibility.
	Seed *int
	// Images provides input images for editing/variation workflows.
	Images []ImageInput
	// ProviderOptions carries provider-specific options keyed by provider name.
	ProviderOptions map[string]any
}

// ImageInput represents an input image for editing workflows.
type ImageInput struct {
	// Data holds inline binary image data.
	Data []byte
	// MimeType is the MIME type of the image (e.g. "image/png").
	MimeType string
	// URL is a URL reference to the image.
	URL string
}

// GenerateImageResult holds the output of an image generation call.
type GenerateImageResult struct {
	// Images holds the generated images.
	Images []GeneratedImage
	// Usage holds token usage information (if available).
	Usage *Usage
	// Warnings holds non-fatal advisories from the provider.
	Warnings []Warning
}

// GeneratedImage holds a single generated image.
type GeneratedImage struct {
	// Data is the raw image bytes.
	Data []byte
	// MimeType is the MIME type of the image (e.g. "image/png").
	MimeType string
}

// Base64 returns the base64-encoded image data.
func (g GeneratedImage) Base64() string {
	return base64.StdEncoding.EncodeToString(g.Data)
}

// Bytes returns the raw image bytes.
func (g GeneratedImage) Bytes() []byte {
	return g.Data
}
