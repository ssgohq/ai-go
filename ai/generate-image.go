package ai

import (
	"context"
	"errors"
)

// GenerateImage generates images using the specified ImageModel.
// It delegates to the model's Generate method after basic validation.
func GenerateImage(ctx context.Context, req GenerateImageRequest) (*GenerateImageResult, error) {
	if req.Model == nil {
		return nil, errors.New("ai: GenerateImage requires a non-nil Model")
	}
	return req.Model.Generate(ctx, req)
}
