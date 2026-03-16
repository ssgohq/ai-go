package uistream

import (
	"fmt"

	"github.com/open-ai-sdk/ai-go/ai"
)

// ToGenerateTextRequest converts a ChatRequestEnvelope (from an HTTP body)
// into an ai.GenerateTextRequest ready for ai.GenerateText or ai.StreamText.
//
// Body hints recognised (all optional):
//   - "system"    string  — system prompt
//   - "maxSteps"  float64 — maximum tool-loop iterations (JSON numbers are float64)
//   - "maxTokens" float64 — max completion tokens
func ToGenerateTextRequest(envelope ChatRequestEnvelope, model ai.LanguageModel) ai.GenerateTextRequest {
	req := ai.GenerateTextRequest{
		Model:    model,
		Messages: ToAIMessages(envelope.Messages),
	}

	applyBodyHints(&req, envelope.Body)

	return req
}

// ToGenerateTextRequestFromRegistry resolves the model from the registry using
// envelope.Body["modelId"] (falling back to the registry's default prefix) and
// then delegates to ToGenerateTextRequest.
func ToGenerateTextRequestFromRegistry(
	envelope ChatRequestEnvelope,
	registry *ai.Registry,
) (ai.GenerateTextRequest, error) {
	modelID, _ := envelope.Body["modelId"].(string) //nolint:errcheck // type assertion ok
	if modelID == "" {
		return ai.GenerateTextRequest{}, fmt.Errorf("envelope-converter: Body[\"modelId\"] is missing or empty")
	}

	model, err := registry.Model(modelID)
	if err != nil {
		return ai.GenerateTextRequest{}, fmt.Errorf("envelope-converter: %w", err)
	}

	return ToGenerateTextRequest(envelope, model), nil
}

// applyBodyHints reads well-known keys from the body map and populates the request.
func applyBodyHints(req *ai.GenerateTextRequest, body map[string]any) {
	if body == nil {
		return
	}

	if system, ok := body["system"].(string); ok && system != "" {
		req.System = system
	}

	if maxSteps, ok := numericInt(body["maxSteps"]); ok && maxSteps > 0 {
		req.MaxSteps = maxSteps
	}

	if maxTokens, ok := numericInt(body["maxTokens"]); ok && maxTokens > 0 {
		req.Settings.MaxTokens = maxTokens
	}
}

// numericInt converts common numeric JSON types to int.
func numericInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	}
	return 0, false
}
