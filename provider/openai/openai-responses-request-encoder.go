package openai

import (
	"fmt"
	"strings"

	"github.com/open-ai-sdk/ai-go/ai"
)

// responsesRequest is the JSON body sent to the OpenAI Responses API POST /v1/responses.
type responsesRequest struct {
	Model              string            `json:"model"`
	Input              []inputItem       `json:"input"`
	Stream             bool              `json:"stream,omitempty"`
	MaxOutputTokens    int               `json:"max_output_tokens,omitempty"`
	Temperature        *float32          `json:"temperature,omitempty"`
	TopP               *float32          `json:"top_p,omitempty"`
	PreviousResponseID string            `json:"previous_response_id,omitempty"`
	Reasoning          *reasoningConfig  `json:"reasoning,omitempty"`
	Tools              []responsesTool   `json:"tools,omitempty"`
	Text               *textConfig       `json:"text,omitempty"`
	Store              *bool             `json:"store,omitempty"`
	User               string            `json:"user,omitempty"`
	Metadata           map[string]string `json:"metadata,omitempty"`
	Include            []string          `json:"include,omitempty"`
}

type reasoningConfig struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type textConfig struct {
	Format *textFormat `json:"format,omitempty"`
}

type textFormat struct {
	Type       string         `json:"type"` // "text" or "json_schema"
	JSONSchema *jsonSchemaRef `json:"json_schema,omitempty"`
}

type jsonSchemaRef struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
	Strict bool           `json:"strict"`
}

// inputItem is a union type for all Responses API input items.
type inputItem struct {
	// Role-based messages.
	Role    string      `json:"role,omitempty"`
	Content []inputPart `json:"content,omitempty"`

	// Function call (assistant tool use).
	Type      string `json:"type,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// Function call output (tool result).
	Output string `json:"output,omitempty"`
}

// inputPart is a single part inside a user or assistant message.
type inputPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
}

// responsesTool describes a tool available to the model.
type responsesTool struct {
	Type        string         `json:"type"` // "function" or built-in name
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// encodeRequest builds a responsesRequest from an ai.LanguageModelRequest.
func encodeRequest(modelID string, req ai.LanguageModelRequest, stream bool) (responsesRequest, []ai.Warning, error) {
	opts := parseProviderOptions(req.ProviderOptions)
	var warnings []ai.Warning

	input, encWarnings, err := encodeInput(req)
	if err != nil {
		return responsesRequest{}, nil, err
	}
	warnings = append(warnings, encWarnings...)

	r := responsesRequest{
		Model:  modelID,
		Input:  input,
		Stream: stream,
	}

	// Token limit: provider option takes precedence over settings.
	if opts.MaxOutputTokens > 0 {
		r.MaxOutputTokens = opts.MaxOutputTokens
	} else if req.Settings.MaxTokens > 0 {
		r.MaxOutputTokens = req.Settings.MaxTokens
	}

	if req.Settings.Temperature != nil {
		r.Temperature = req.Settings.Temperature
	}
	if req.Settings.TopP != nil {
		r.TopP = req.Settings.TopP
	}

	if len(req.Settings.StopSequences) > 0 {
		warnings = append(warnings, ai.Warning{
			Type:    "unsupported-setting",
			Setting: "stopSequences",
			Message: "stopSequences is not supported by the OpenAI Responses API",
		})
	}

	if opts.PreviousResponseID != "" {
		r.PreviousResponseID = opts.PreviousResponseID
	}

	// Reasoning settings for o-series and gpt-5 reasoning models.
	if opts.ReasoningEffort != "" || opts.ReasoningSummary != "" {
		r.Reasoning = &reasoningConfig{
			Effort:  opts.ReasoningEffort,
			Summary: opts.ReasoningSummary,
		}
	}

	if opts.User != "" {
		r.User = opts.User
	}
	if opts.Metadata != nil {
		r.Metadata = opts.Metadata
	}
	if opts.Store != nil {
		r.Store = opts.Store
	}

	// Tools: function tools + optional built-in web search.
	tools, toolWarnings := encodeTools(req.Tools, opts)
	warnings = append(warnings, toolWarnings...)
	r.Tools = tools

	// Include list: add sources when web search + IncludeSources requested.
	if opts.EnableWebSearch && opts.IncludeSources {
		r.Include = append(r.Include, "web_search_call.action.sources")
	}

	// Structured output schema.
	if req.Output != nil && req.Output.Type != "text" {
		r.Text = encodeOutputSchema(req.Output)
	}

	return r, warnings, nil
}

// encodeInput converts system prompt + messages to Responses API input items.
func encodeInput(req ai.LanguageModelRequest) ([]inputItem, []ai.Warning, error) {
	var items []inputItem
	var warnings []ai.Warning

	if req.System != "" {
		items = append(items, inputItem{
			Role:    "system",
			Content: []inputPart{{Type: "input_text", Text: req.System}},
		})
	}

	for _, m := range req.Messages {
		encoded, w, err := encodeMessage(m)
		if err != nil {
			return nil, nil, err
		}
		warnings = append(warnings, w...)
		items = append(items, encoded...)
	}
	return items, warnings, nil
}

func encodeMessage(m ai.Message) ([]inputItem, []ai.Warning, error) {
	switch m.Role {
	case ai.RoleUser:
		return encodeUserMessage(m)
	case ai.RoleAssistant:
		return encodeAssistantMessage(m)
	case ai.RoleTool:
		return encodeToolResultMessage(m)
	default:
		return nil, nil, fmt.Errorf("openai: unsupported message role %q", m.Role)
	}
}

func encodeUserMessage(m ai.Message) ([]inputItem, []ai.Warning, error) {
	var parts []inputPart
	var warnings []ai.Warning

	for _, p := range m.Content {
		switch p.Type {
		case ai.ContentPartTypeText:
			parts = append(parts, inputPart{Type: "input_text", Text: p.Text})

		case ai.ContentPartTypeImageURL:
			url := p.ImageURL
			if strings.HasPrefix(url, "file-") {
				// Treat as OpenAI file ID.
				parts = append(parts, inputPart{Type: "input_image", FileID: url})
			} else {
				parts = append(parts, inputPart{Type: "input_image", ImageURL: url})
			}

		case ai.ContentPartTypeFile:
			url := p.FileURL
			if strings.HasPrefix(url, "file-") {
				// OpenAI file ID.
				parts = append(parts, inputPart{Type: "input_file", FileID: url})
			} else {
				parts = append(parts, inputPart{Type: "input_file", FileURL: url})
			}

		default:
			warnings = append(warnings, ai.Warning{
				Type:    "unsupported-setting",
				Setting: string(p.Type),
				Message: fmt.Sprintf("openai: unsupported user content part type %q, skipping", p.Type),
			})
		}
	}

	if len(parts) == 0 {
		return nil, warnings, nil
	}
	return []inputItem{{Role: "user", Content: parts}}, warnings, nil
}

func encodeAssistantMessage(m ai.Message) ([]inputItem, []ai.Warning, error) {
	var items []inputItem

	var textParts []inputPart
	for _, p := range m.Content {
		switch p.Type {
		case ai.ContentPartTypeText:
			textParts = append(textParts, inputPart{Type: "output_text", Text: p.Text})
		case ai.ContentPartTypeToolCall:
			// Flush any accumulated text first.
			if len(textParts) > 0 {
				items = append(items, inputItem{Role: "assistant", Content: textParts})
				textParts = nil
			}
			items = append(items, inputItem{
				Type:      "function_call",
				CallID:    p.ToolCallID,
				Name:      p.ToolCallName,
				Arguments: string(p.ToolCallArgs),
			})
		}
	}
	if len(textParts) > 0 {
		items = append(items, inputItem{Role: "assistant", Content: textParts})
	}
	return items, nil, nil
}

func encodeToolResultMessage(m ai.Message) ([]inputItem, []ai.Warning, error) {
	var items []inputItem
	for _, p := range m.Content {
		if p.Type == ai.ContentPartTypeToolResult {
			items = append(items, inputItem{
				Type:   "function_call_output",
				CallID: p.ToolResultID,
				Output: p.ToolResultOutput,
			})
		}
	}
	return items, nil, nil
}

func encodeTools(defs []ai.ToolDefinition, opts ProviderOptions) ([]responsesTool, []ai.Warning) {
	var tools []responsesTool

	for _, d := range defs {
		tools = append(tools, responsesTool{
			Type:        "function",
			Name:        d.Name,
			Description: d.Description,
			Parameters:  d.InputSchema,
		})
	}

	if opts.EnableWebSearch {
		tools = append(tools, responsesTool{Type: "web_search_preview"})
	}

	return tools, nil
}

func encodeOutputSchema(o *ai.OutputSchema) *textConfig {
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
	return &textConfig{
		Format: &textFormat{
			Type: "json_schema",
			JSONSchema: &jsonSchemaRef{
				Name:   "structured_output",
				Schema: schema,
				Strict: true,
			},
		},
	}
}
