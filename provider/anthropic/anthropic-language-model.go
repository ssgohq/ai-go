package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

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
	// Use Transport.ResponseHeaderTimeout for initial handshake protection
	// instead of Client.Timeout, which would kill long-running SSE streams.
	transport := &http.Transport{
		ResponseHeaderTimeout: cfg.Timeout,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
	}
	return &LanguageModel{
		modelID: modelID,
		config:  cfg,
		client:  &http.Client{Transport: transport},
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
		respBody, readErr := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
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
	Model      string               `json:"model"`
	MaxTokens  int                  `json:"max_tokens"`
	System     string               `json:"system,omitempty"`
	Messages   []anthropicMsg       `json:"messages"`
	Stream     bool                 `json:"stream"`
	Tools      []anthropicTool      `json:"tools,omitempty"`
	ToolChoice *anthropicToolChoice `json:"tool_choice,omitempty"`
	Thinking   *thinkingConfig      `json:"thinking,omitempty"`
}

type anthropicToolChoice struct {
	Type string `json:"type"`           // "auto", "any", "tool", "none"
	Name string `json:"name,omitempty"` // set when Type == "tool"
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
	if req.Output != nil {
		return nil, fmt.Errorf("anthropic: output schema is not yet supported")
	}

	ar := anthropicRequest{
		Model:     m.modelID,
		MaxTokens: req.Settings.MaxTokens,
		Stream:    true,
	}
	if ar.MaxTokens == 0 {
		ar.MaxTokens = 8192
	}

	ar.System = req.System
	ar.ToolChoice = mapToolChoice(req.ToolChoice)
	ar.Thinking = extractThinkingConfig(req.ProviderOptions)
	ar.Messages = encodeMessages(req.Messages)
	ar.Tools = encodeTools(req.Tools)

	// Enable caching on last tool if caching is enabled
	if m.config.EnableCaching && len(ar.Tools) > 0 {
		ar.Tools[len(ar.Tools)-1].CacheControl = &cacheControl{Type: "ephemeral"}
	}

	return json.Marshal(ar)
}

func mapToolChoice(tc *ai.ToolChoice) *anthropicToolChoice {
	if tc == nil {
		return nil
	}
	switch tc.Type {
	case "auto":
		return &anthropicToolChoice{Type: "auto"}
	case "none":
		return &anthropicToolChoice{Type: "none"}
	case "required":
		return &anthropicToolChoice{Type: "any"}
	case "tool":
		return &anthropicToolChoice{Type: "tool", Name: tc.ToolName}
	default:
		return nil
	}
}

func extractThinkingConfig(opts map[string]any) *thinkingConfig {
	anthOpts, ok := opts["anthropic"]
	if !ok {
		return nil
	}
	om, ok := anthOpts.(map[string]any)
	if !ok {
		return nil
	}
	thinking, ok := om["thinking"].(bool)
	if !ok || !thinking {
		return nil
	}
	budget := 10000
	if b, ok := om["thinkingBudget"].(int); ok {
		budget = b
	}
	return &thinkingConfig{Type: "enabled", BudgetTokens: budget}
}

func encodeMessages(msgs []ai.Message) []anthropicMsg {
	var out []anthropicMsg
	for _, msg := range msgs {
		am := anthropicMsg{Role: string(msg.Role)}
		for _, part := range msg.Content {
			if cb := encodeContentPart(part); cb != nil {
				am.Content = append(am.Content, *cb)
			}
		}
		if len(am.Content) > 0 {
			out = append(out, am)
		}
	}
	return out
}

func encodeContentPart(part ai.ContentPart) *contentBlock {
	switch part.Type {
	case ai.ContentPartTypeText:
		return &contentBlock{Type: "text", Text: part.Text}
	case ai.ContentPartTypeImageURL:
		if len(part.Data) == 0 {
			return nil
		}
		return &contentBlock{
			Type: "image",
			Source: &imageSource{
				Type:      "base64",
				MediaType: part.MimeType,
				Data:      base64.StdEncoding.EncodeToString(part.Data),
			},
		}
	case ai.ContentPartTypeToolCall:
		return &contentBlock{
			Type:  "tool_use",
			ID:    part.ToolCallID,
			Name:  part.ToolCallName,
			Input: part.ToolCallArgs,
		}
	case ai.ContentPartTypeToolResult:
		return &contentBlock{
			Type:      "tool_result",
			ToolUseID: part.ToolResultID,
			Content:   part.ToolResultOutput,
		}
	case ai.ContentPartTypeReasoning:
		return &contentBlock{
			Type:      "thinking",
			Thinking:  part.ReasoningText,
			Signature: part.ThoughtSignature,
		}
	default:
		return nil
	}
}

func encodeTools(tools []ai.ToolDefinition) []anthropicTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropicTool, len(tools))
	for i, tool := range tools {
		out[i] = anthropicTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
	}
	return out
}
