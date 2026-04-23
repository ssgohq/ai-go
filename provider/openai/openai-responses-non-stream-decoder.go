package openai

import (
	"encoding/json"
	"fmt"

	"github.com/open-ai-sdk/ai-go/ai"
)

// responsesNonStreamResponse is the JSON body returned by a non-streaming
// POST /v1/responses call.
type responsesNonStreamResponse struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Output []struct {
		Type    string `json:"type"`
		Content []struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Annotations []any  `json:"annotations"`
		} `json:"content"`
		// function_call fields
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"output"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// decodeResponsesNonStream converts a raw non-streaming Responses API JSON body
// into an ai.GenerateTextResult.
func decodeResponsesNonStream(body []byte, warnings []ai.Warning) (*ai.GenerateTextResult, error) {
	var resp responsesNonStreamResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("openai: %s: %s", resp.Error.Code, resp.Error.Message)
	}

	result := &ai.GenerateTextResult{
		Warnings: warnings,
	}

	// Aggregate text and tool calls from output items.
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					result.Text += part.Text
				}
			}
		case "function_call":
			result.Steps = appendToolCall(result.Steps, ai.ToolCallOutput{
				ID:   item.CallID,
				Name: item.Name,
				Args: json.RawMessage(item.Arguments),
			})
		}
	}

	if len(result.Steps) == 0 && result.Text != "" {
		result.Steps = []ai.StepOutput{{Text: result.Text}}
	} else if len(result.Steps) > 0 && result.Text != "" {
		result.Steps[len(result.Steps)-1].Text = result.Text
	}

	if resp.Usage != nil {
		result.TotalUsage = ai.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	hasToolCalls := len(result.Steps) > 0 && len(result.Steps[0].ToolCalls) > 0
	result.FinishReason = mapResponsesFinishReason(resp.Status, hasToolCalls)
	result.RawFinishReason = resp.Status
	result.ProviderMetadata = map[string]any{
		"openai": map[string]any{
			"responseId": resp.ID,
		},
	}
	for i := range result.Steps {
		result.Steps[i].FinishReason = result.FinishReason
		result.Steps[i].RawFinishReason = result.RawFinishReason
		result.Steps[i].ProviderMetadata = result.ProviderMetadata
		result.Steps[i].Response = ai.Response{Messages: ai.ResponseMessagesForStep(result.Steps[i], nil)}
	}
	result.Response = ai.Response{Messages: ai.ResponseMessagesForSteps(result.Steps, nil)}

	return result, nil
}

// appendToolCall ensures at least one StepOutput exists and appends the tool call to it.
func appendToolCall(steps []ai.StepOutput, tc ai.ToolCallOutput) []ai.StepOutput {
	if len(steps) == 0 {
		steps = []ai.StepOutput{{}}
	}
	steps[0].ToolCalls = append(steps[0].ToolCalls, tc)
	return steps
}
