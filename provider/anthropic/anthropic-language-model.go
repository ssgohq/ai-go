package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/open-ai-sdk/ai-go/ai"
)

// LanguageModel implements ai.LanguageModel using the Anthropic Messages API.
type LanguageModel struct {
	modelID string
	config  Config
	client  *http.Client
}

// NewLanguageModel creates a native Anthropic language model.
func NewLanguageModel(modelID string, cfg Config) *LanguageModel {
	cfg = cfg.withDefaults()
	return &LanguageModel{
		modelID: modelID,
		config:  cfg,
		client:  &http.Client{Timeout: cfg.Timeout},
	}
}

// ModelID returns the model identifier.
func (m *LanguageModel) ModelID() string { return m.modelID }

// Stream sends a streaming request to the Anthropic Messages API.
func (m *LanguageModel) Stream(ctx context.Context, req ai.LanguageModelRequest) (<-chan ai.StreamEvent, error) {
	body, err := m.encodeRequest(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: encode request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		m.config.BaseURL+"/v1/messages",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("anthropic: build http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-API-Key", m.config.APIKey)
	httpReq.Header.Set("Anthropic-Version", m.config.APIVersion)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: http request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf(
				"anthropic: unexpected status %d (failed to read body: %w)",
				resp.StatusCode, readErr,
			)
		}
		return nil, fmt.Errorf("anthropic: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan ai.StreamEvent, 64)
	go decodeSSEStream(ctx, resp.Body, ch)
	return ch, nil
}

// anthropicRequest is the Messages API request body.
type anthropicRequest struct {
	Model     string          `json:"model"`
	MaxTokens int             `json:"max_tokens"`
	System    string          `json:"system,omitempty"`
	Messages  []anthropicMsg  `json:"messages"`
	Stream    bool            `json:"stream"`
	Tools     []anthropicTool `json:"tools,omitempty"`
	Thinking  *thinkingConfig `json:"thinking,omitempty"`
}

type anthropicMsg struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type         string          `json:"type"`
	Text         string          `json:"text,omitempty"`
	Source       *imageSource    `json:"source,omitempty"`
	ID           string          `json:"id,omitempty"`
	Name         string          `json:"name,omitempty"`
	Input        json.RawMessage `json:"input,omitempty"`
	ToolUseID    string          `json:"tool_use_id,omitempty"`
	Content      string          `json:"content,omitempty"`
	CacheControl *cacheControl   `json:"cache_control,omitempty"`
	Thinking     string          `json:"thinking,omitempty"`
	Signature    string          `json:"signature,omitempty"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type cacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

type anthropicTool struct {
	Name         string         `json:"name"`
	Description  string         `json:"description,omitempty"`
	InputSchema  map[string]any `json:"input_schema"`
	CacheControl *cacheControl  `json:"cache_control,omitempty"`
}

type thinkingConfig struct {
	Type         string `json:"type"` // "enabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

func (m *LanguageModel) encodeRequest(req ai.LanguageModelRequest) ([]byte, error) {
	ar := anthropicRequest{
		Model:     m.modelID,
		MaxTokens: req.Settings.MaxTokens,
		Stream:    true,
	}
	if ar.MaxTokens == 0 {
		ar.MaxTokens = 8192
	}

	// System prompt
	ar.System = req.System

	// Check for extended thinking in provider options
	if opts, ok := req.ProviderOptions["anthropic"]; ok {
		if om, ok := opts.(map[string]any); ok {
			if thinking, ok := om["thinking"].(bool); ok && thinking {
				budget := 10000
				if b, ok := om["thinkingBudget"].(int); ok {
					budget = b
				}
				ar.Thinking = &thinkingConfig{Type: "enabled", BudgetTokens: budget}
			}
		}
	}

	// Convert messages
	for _, msg := range req.Messages {
		am := anthropicMsg{Role: string(msg.Role)}
		for _, part := range msg.Content {
			switch part.Type {
			case ai.ContentPartTypeText:
				am.Content = append(am.Content, contentBlock{Type: "text", Text: part.Text})
			case ai.ContentPartTypeImageURL:
				if len(part.Data) > 0 {
					am.Content = append(am.Content, contentBlock{
						Type: "image",
						Source: &imageSource{
							Type:      "base64",
							MediaType: part.MimeType,
							Data:      string(part.Data), // needs base64 encoding
						},
					})
				}
			case ai.ContentPartTypeToolCall:
				am.Content = append(am.Content, contentBlock{
					Type:  "tool_use",
					ID:    part.ToolCallID,
					Name:  part.ToolCallName,
					Input: part.ToolCallArgs,
				})
			case ai.ContentPartTypeToolResult:
				am.Content = append(am.Content, contentBlock{
					Type:      "tool_result",
					ToolUseID: part.ToolResultID,
					Content:   part.ToolResultOutput,
				})
			case ai.ContentPartTypeReasoning:
				am.Content = append(am.Content, contentBlock{
					Type:      "thinking",
					Thinking:  part.ReasoningText,
					Signature: part.ThoughtSignature,
				})
			}
		}
		if len(am.Content) > 0 {
			ar.Messages = append(ar.Messages, am)
		}
	}

	// Convert tools
	for _, tool := range req.Tools {
		at := anthropicTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
		// Enable caching on last tool if caching is enabled
		if m.config.EnableCaching {
			at.CacheControl = &cacheControl{Type: "ephemeral"}
		}
		ar.Tools = append(ar.Tools, at)
	}

	return json.Marshal(ar)
}
