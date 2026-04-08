package openaichat

import (
	"encoding/base64"
	"fmt"

	"github.com/open-ai-sdk/ai-go/ai"
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
	ToolChoice     any              `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat  `json:"response_format,omitempty"`
	Stop           []string         `json:"stop,omitempty"`
	Seed           *int             `json:"seed,omitempty"`
	TopP           float32          `json:"top_p,omitempty"`
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
	// ExtraTools holds additional provider-specific tool entries to append after
	// encoding the standard function tools. Each entry is a raw JSON-serializable
	// map, e.g. map[string]any{"type": "google_search"} for Gemini grounding.
	ExtraTools []map[string]any
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
	if req.Settings.TopP != nil {
		cr.TopP = *req.Settings.TopP
	}
	if req.Settings.Seed != nil {
		cr.Seed = req.Settings.Seed
	}
	if len(req.Settings.StopSequences) > 0 {
		cr.Stop = req.Settings.StopSequences
	}

	if len(req.Tools) > 0 || len(params.ExtraTools) > 0 {
		toolDefs := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			toolDefs[i] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.InputSchema,
				},
			}
		}
		if params.SanitizeTools != nil {
			toolDefs = params.SanitizeTools(toolDefs)
		}
		toolDefs = append(toolDefs, params.ExtraTools...)
		cr.Tools = toolDefs
		if len(req.Tools) > 0 {
			cr.ToolChoice = encodeToolChoice(req.ToolChoice)
		}
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
			var imageURL string
			switch {
			case len(part.Data) > 0:
				mimeType := part.MimeType
				if mimeType == "" {
					mimeType = "image/png"
				}
				imageURL = "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(part.Data)
			default:
				imageURL = part.ImageURL
			}
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": imageURL},
			})
		case ai.ContentPartTypeFile:
			// Chat completions API does not support file IDs or binary uploads natively.
			// Encode inline data as a data: URI stub or fall back to URL text.
			switch {
			case len(part.Data) > 0:
				mimeType := part.MimeType
				if mimeType == "" {
					mimeType = "application/octet-stream"
				}
				dataURI := "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(part.Data)
				parts = append(parts, map[string]any{
					"type": "text",
					"text": fmt.Sprintf("[file: %s]", dataURI),
				})
			default:
				parts = append(parts, map[string]any{
					"type": "text",
					"text": fmt.Sprintf("[file url: %s]", part.FileURL),
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
			if part.ThoughtSignature != "" {
				call["extra_content"] = map[string]any{
					"google": map[string]any{
						"thought_signature": part.ThoughtSignature,
					},
				}
			}
			toolCalls = append(toolCalls, call)
		case ai.ContentPartTypeReasoning:
			// Reasoning parts are provider-specific; most OpenAI-compatible APIs do not
			// accept them as message content. Omit silently to maintain compatibility.
		}
	}

	msg := map[string]any{"role": string(m.Role)}
	if len(parts) > 0 {
		msg["content"] = parts
	} else if m.Role == ai.RoleAssistant {
		msg["content"] = nil
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

// encodeToolChoice converts an ai.ToolChoice to the OpenAI wire format.
// nil / "auto" → "auto" (string), "none" → "none", "required" → "required",
// "tool" → object {"type":"function","function":{"name":"..."}}.
func encodeToolChoice(tc *ai.ToolChoice) any {
	if tc == nil {
		return "auto"
	}
	switch tc.Type {
	case "none":
		return "none"
	case "required":
		return "required"
	case "tool":
		return map[string]any{
			"type":     "function",
			"function": map[string]any{"name": tc.ToolName},
		}
	default:
		return "auto"
	}
}

// encodeOutputSchema converts an ai.OutputSchema to OpenAI response_format.
// "json_object" → {type: "json_object"} (no schema).
// "object" / "array" → {type: "json_schema", json_schema: {...}}.
func encodeOutputSchema(o *ai.OutputSchema) *ResponseFormat {
	if o.Type == "json_object" {
		return &ResponseFormat{Type: "json_object"}
	}
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
