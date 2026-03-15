package openaichat

import (
	"fmt"
	"strings"

	"github.com/ssgohq/ai-go/ai"
)

// ChatRequest is the JSON body for an OpenAI-style chat completions request.
type ChatRequest struct {
	Model          string           `json:"model"`
	Messages       []map[string]any `json:"messages"`
	Stream         bool             `json:"stream"`
	StreamOptions  map[string]any   `json:"stream_options,omitempty"`
	Temperature    float32          `json:"temperature,omitempty"`
	MaxTokens      int              `json:"max_tokens,omitempty"`
	Tools          []map[string]any `json:"tools,omitempty"`
	ToolChoice     string           `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat  `json:"response_format,omitempty"`
	Stop           []string         `json:"stop,omitempty"`
}

// ResponseFormat encodes the response_format field.
type ResponseFormat struct {
	Type       string         `json:"type"`
	JSONSchema *JSONSchemaRef `json:"json_schema,omitempty"`
}

// JSONSchemaRef wraps a named JSON schema for structured output.
type JSONSchemaRef struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
	Strict bool           `json:"strict"`
}

// EncodeRequestParams carries configuration for encoding a single request.
type EncodeRequestParams struct {
	ModelID string
	// SanitizeTools is an optional hook to transform tool schemas before sending.
	// Gemini uses this to strip unsupported JSON Schema keys.
	SanitizeTools func(tools []map[string]any) []map[string]any
	// IncludeStreamUsage enables stream_options.include_usage for streaming requests.
	IncludeStreamUsage bool
}

// EncodeRequest converts an ai.LanguageModelRequest into a ChatRequest.
func EncodeRequest(
	params EncodeRequestParams,
	req ai.LanguageModelRequest,
	streaming bool,
) (ChatRequest, error) {
	msgs, err := encodeMessages(req.System, req.Messages)
	if err != nil {
		return ChatRequest{}, err
	}

	cr := ChatRequest{
		Model:    params.ModelID,
		Messages: msgs,
		Stream:   streaming,
	}

	if streaming && params.IncludeStreamUsage {
		cr.StreamOptions = map[string]any{"include_usage": true}
	}

	if req.Settings.MaxTokens > 0 {
		cr.MaxTokens = req.Settings.MaxTokens
	}
	if req.Settings.Temperature != nil {
		cr.Temperature = *req.Settings.Temperature
	}

	if len(req.Tools) > 0 {
		toolDefs := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			toolDefs[i] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.Parameters,
				},
			}
		}
		if params.SanitizeTools != nil {
			toolDefs = params.SanitizeTools(toolDefs)
		}
		cr.Tools = toolDefs
		cr.ToolChoice = "auto"
	}

	if req.Output != nil && req.Output.Type != "text" {
		cr.ResponseFormat = encodeOutputSchema(req.Output)
	}

	return cr, nil
}

func encodeMessages(system string, messages []ai.Message) ([]map[string]any, error) {
	var out []map[string]any

	if system != "" {
		out = append(out, map[string]any{"role": "system", "content": system})
	}

	for _, m := range messages {
		encoded, err := encodeMessage(m)
		if err != nil {
			return nil, err
		}
		out = append(out, encoded)
	}
	return out, nil
}

func encodeMessage(m ai.Message) (map[string]any, error) {
	switch m.Role {
	case ai.RoleTool:
		return encodeToolResultMessage(m)
	default:
		return encodeContentMessage(m)
	}
}

func encodeContentMessage(m ai.Message) (map[string]any, error) {
	// Single text part shortcut.
	if len(m.Content) == 1 && m.Content[0].Type == ai.ContentPartTypeText {
		return map[string]any{"role": string(m.Role), "content": m.Content[0].Text}, nil
	}

	parts := make([]map[string]any, 0, len(m.Content))
	var toolCalls []map[string]any

	for _, part := range m.Content {
		switch part.Type {
		case ai.ContentPartTypeText:
			parts = append(parts, map[string]any{"type": "text", "text": part.Text})
		case ai.ContentPartTypeImageURL:
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": part.ImageURL},
			})
		case ai.ContentPartTypeFile:
			url := part.FileURL
			if strings.HasPrefix(url, "data:") {
				parts = append(parts, map[string]any{
					"type": "text",
					"text": fmt.Sprintf("[file: %s]", part.MimeType),
				})
			} else {
				parts = append(parts, map[string]any{
					"type": "text",
					"text": fmt.Sprintf("[file url: %s]", url),
				})
			}
		case ai.ContentPartTypeToolCall:
			call := map[string]any{
				"id":   part.ToolCallID,
				"type": "function",
				"function": map[string]string{
					"name":      part.ToolCallName,
					"arguments": string(part.ToolCallArgs),
				},
			}
			toolCalls = append(toolCalls, call)
		}
	}

	msg := map[string]any{"role": string(m.Role)}
	if len(parts) > 0 {
		msg["content"] = parts
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	return msg, nil
}

func encodeToolResultMessage(m ai.Message) (map[string]any, error) {
	for _, part := range m.Content {
		if part.Type == ai.ContentPartTypeToolResult {
			return map[string]any{
				"role":         "tool",
				"tool_call_id": part.ToolResultID,
				"content":      part.ToolResultOutput,
			}, nil
		}
	}
	return map[string]any{"role": "tool", "content": ""}, nil
}

func encodeOutputSchema(o *ai.OutputSchema) *ResponseFormat {
	schema := o.Schema
	if o.Type == "object" && schema != nil {
		if _, ok := schema["type"]; !ok {
			wrapped := make(map[string]any, len(schema)+1)
			wrapped["type"] = "object"
			for k, v := range schema {
				wrapped[k] = v
			}
			schema = wrapped
		}
	}
	return &ResponseFormat{
		Type: "json_schema",
		JSONSchema: &JSONSchemaRef{
			Name:   "structured_output",
			Schema: schema,
			Strict: true,
		},
	}
}
