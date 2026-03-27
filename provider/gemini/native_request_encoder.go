package gemini

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/open-ai-sdk/ai-go/ai"
)

// nativeRequest is the top-level JSON body for :streamGenerateContent.
type nativeRequest struct {
	Contents          []nativeContent          `json:"contents"`
	SystemInstruction *nativeSystemInstruction `json:"systemInstruction,omitempty"`
	GenerationConfig  *nativeGenerationConfig  `json:"generationConfig,omitempty"`
	SafetySettings    []any                    `json:"safetySettings,omitempty"`
	Tools             []any                    `json:"tools,omitempty"`
	ToolConfig        any                      `json:"toolConfig,omitempty"`
	CachedContent     string                   `json:"cachedContent,omitempty"`
	Labels            map[string]string        `json:"labels,omitempty"`
}

type nativeContent struct {
	Role  string       `json:"role"`
	Parts []nativePart `json:"parts"`
}

type nativePart struct {
	Text             string            `json:"text,omitempty"`
	Thought          *bool             `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
	InlineData       *nativeInlineData `json:"inlineData,omitempty"`
	FileData         *nativeFileData   `json:"fileData,omitempty"`
	FunctionCall     *nativeFuncCall   `json:"functionCall,omitempty"`
	FunctionResponse *nativeFuncResp   `json:"functionResponse,omitempty"`
}

type nativeInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type nativeFileData struct {
	MimeType string `json:"mimeType"`
	FileUri  string `json:"fileUri"`
}

type nativeFuncCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type nativeFuncResp struct {
	Name     string `json:"name"`
	Response any    `json:"response"`
}

type nativeSystemInstruction struct {
	Parts []nativeTextPart `json:"parts"`
}

type nativeTextPart struct {
	Text string `json:"text"`
}

type nativeGenerationConfig struct {
	MaxOutputTokens    *int                `json:"maxOutputTokens,omitempty"`
	Temperature        *float32            `json:"temperature,omitempty"`
	TopK               *int                `json:"topK,omitempty"`
	TopP               *float32            `json:"topP,omitempty"`
	StopSequences      []string            `json:"stopSequences,omitempty"`
	Seed               *int                `json:"seed,omitempty"`
	ResponseMimeType   string              `json:"responseMimeType,omitempty"`
	ResponseSchema     map[string]any      `json:"responseSchema,omitempty"`
	ThinkingConfig     *nativeThinkingCfg  `json:"thinkingConfig,omitempty"`
	ResponseModalities []string            `json:"responseModalities,omitempty"`
	ImageConfig        *nativeImageConfig  `json:"imageConfig,omitempty"`
}

type nativeThinkingCfg struct {
	ThinkingBudget  *int   `json:"thinkingBudget,omitempty"`
	IncludeThoughts *bool  `json:"includeThoughts,omitempty"`
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
}

type nativeImageConfig struct {
	AspectRatio string `json:"aspectRatio,omitempty"`
	ImageSize   string `json:"imageSize,omitempty"`
}

// encodeNativeRequest converts an ai.LanguageModelRequest to the native Gemini
// request body for the :streamGenerateContent endpoint.
func encodeNativeRequest(req ai.LanguageModelRequest) nativeRequest {
	nr := nativeRequest{}

	// System instruction: collect from req.System and any leading system messages.
	nr.SystemInstruction = buildSystemInstruction(req)

	// Convert messages to native contents.
	nr.Contents = buildContents(req.Messages)

	// Generation config.
	nr.GenerationConfig = buildGenerationConfig(req)

	return nr
}

// buildSystemInstruction collects system text from req.System and any leading
// RoleSystem messages in req.Messages.
func buildSystemInstruction(req ai.LanguageModelRequest) *nativeSystemInstruction {
	var parts []nativeTextPart

	if req.System != "" {
		parts = append(parts, nativeTextPart{Text: req.System})
	}

	for _, msg := range req.Messages {
		if msg.Role != ai.RoleSystem {
			break
		}
		for _, p := range msg.Content {
			if p.Type == ai.ContentPartTypeText && p.Text != "" {
				parts = append(parts, nativeTextPart{Text: p.Text})
			}
		}
	}

	if len(parts) == 0 {
		return nil
	}
	return &nativeSystemInstruction{Parts: parts}
}

// buildContents converts the message list to native Gemini contents,
// skipping leading system messages (already handled by buildSystemInstruction).
func buildContents(messages []ai.Message) []nativeContent {
	var contents []nativeContent

	// Skip leading system messages.
	startIdx := 0
	for startIdx < len(messages) && messages[startIdx].Role == ai.RoleSystem {
		startIdx++
	}

	for _, msg := range messages[startIdx:] {
		switch msg.Role {
		case ai.RoleUser:
			contents = append(contents, nativeContent{
				Role:  "user",
				Parts: encodeUserParts(msg.Content),
			})
		case ai.RoleAssistant:
			contents = append(contents, nativeContent{
				Role:  "model",
				Parts: encodeAssistantParts(msg.Content),
			})
		case ai.RoleTool:
			contents = append(contents, nativeContent{
				Role:  "user",
				Parts: encodeToolParts(msg.Content),
			})
		}
	}

	return contents
}

// encodeUserParts converts user content parts to native parts.
func encodeUserParts(parts []ai.ContentPart) []nativePart {
	var out []nativePart
	for _, p := range parts {
		switch p.Type {
		case ai.ContentPartTypeText:
			out = append(out, nativePart{Text: p.Text})
		case ai.ContentPartTypeImageURL:
			out = append(out, encodeImagePart(p))
		case ai.ContentPartTypeFile:
			out = append(out, encodeFilePart(p))
		}
	}
	return out
}

// encodeAssistantParts converts assistant content parts to native parts.
func encodeAssistantParts(parts []ai.ContentPart) []nativePart {
	var out []nativePart
	for _, p := range parts {
		switch p.Type {
		case ai.ContentPartTypeText:
			out = append(out, nativePart{Text: p.Text})
		case ai.ContentPartTypeReasoning:
			np := nativePart{Text: p.ReasoningText, Thought: boolPtr(true)}
			if p.ThoughtSignature != "" {
				np.ThoughtSignature = p.ThoughtSignature
			}
			out = append(out, np)
		case ai.ContentPartTypeToolCall:
			np := nativePart{
				FunctionCall: &nativeFuncCall{
					Name: p.ToolCallName,
					Args: p.ToolCallArgs,
				},
			}
			if p.ThoughtSignature != "" {
				np.ThoughtSignature = p.ThoughtSignature
			}
			out = append(out, np)
		case ai.ContentPartTypeImageURL:
			out = append(out, encodeImagePart(p))
		case ai.ContentPartTypeFile:
			out = append(out, encodeFilePart(p))
		}
	}
	return out
}

// encodeToolParts converts tool-result content parts to native functionResponse parts.
func encodeToolParts(parts []ai.ContentPart) []nativePart {
	var out []nativePart
	for _, p := range parts {
		if p.Type == ai.ContentPartTypeToolResult {
			out = append(out, nativePart{
				FunctionResponse: &nativeFuncResp{
					Name: p.ToolResultName,
					Response: map[string]string{
						"name":    p.ToolResultName,
						"content": p.ToolResultOutput,
					},
				},
			})
		}
	}
	return out
}

// encodeImagePart encodes an image ContentPart as inlineData or fileData.
// It handles both inline binary data (from ImageDataPart) and URL references (from ImageURLPart).
func encodeImagePart(p ai.ContentPart) nativePart {
	if len(p.Data) > 0 {
		return nativePart{
			InlineData: &nativeInlineData{
				MimeType: p.MimeType,
				Data:     base64.StdEncoding.EncodeToString(p.Data),
			},
		}
	}
	return encodeMediaFromURL(p.ImageURL, p.MimeType)
}

// encodeMediaFromURL converts a URL (possibly data: URI) to an inlineData or fileData part.
func encodeMediaFromURL(url, mimeType string) nativePart {
	if strings.HasPrefix(url, "data:") {
		mime, data, ok := parseDataURI(url)
		if ok {
			return nativePart{InlineData: &nativeInlineData{MimeType: mime, Data: data}}
		}
	}
	m := mimeType
	if m == "" {
		m = guessMimeTypeFromURL(url)
	}
	return nativePart{FileData: &nativeFileData{MimeType: m, FileUri: url}}
}

// encodeFilePart encodes a file ContentPart as inlineData or fileData.
func encodeFilePart(p ai.ContentPart) nativePart {
	if len(p.Data) > 0 {
		return nativePart{
			InlineData: &nativeInlineData{
				MimeType: p.MimeType,
				Data:     base64.StdEncoding.EncodeToString(p.Data),
			},
		}
	}
	if strings.HasPrefix(p.FileURL, "data:") {
		mime, data, ok := parseDataURI(p.FileURL)
		if ok {
			return nativePart{InlineData: &nativeInlineData{MimeType: mime, Data: data}}
		}
	}
	m := p.MimeType
	if m == "" {
		m = guessMimeTypeFromURL(p.FileURL)
	}
	return nativePart{FileData: &nativeFileData{MimeType: m, FileUri: p.FileURL}}
}

// buildGenerationConfig constructs the generationConfig from settings, provider options, and output schema.
func buildGenerationConfig(req ai.LanguageModelRequest) *nativeGenerationConfig {
	cfg := &nativeGenerationConfig{}
	empty := true

	s := req.Settings
	if s.MaxTokens > 0 {
		cfg.MaxOutputTokens = &s.MaxTokens
		empty = false
	}
	if s.Temperature != nil {
		cfg.Temperature = s.Temperature
		empty = false
	}
	if s.TopP != nil {
		cfg.TopP = s.TopP
		empty = false
	}
	if s.TopK != nil {
		cfg.TopK = s.TopK
		empty = false
	}
	if s.Seed != nil {
		cfg.Seed = s.Seed
		empty = false
	}
	if len(s.StopSequences) > 0 {
		cfg.StopSequences = s.StopSequences
		empty = false
	}

	// ThinkingConfig from provider options.
	opts := parseProviderOptions(req.ProviderOptions)
	if opts.ThinkingConfig != nil {
		tc := opts.ThinkingConfig
		ntc := &nativeThinkingCfg{}
		tcEmpty := true
		if tc.ThinkingBudget != nil {
			ntc.ThinkingBudget = tc.ThinkingBudget
			tcEmpty = false
		}
		if tc.IncludeThoughts != nil {
			ntc.IncludeThoughts = tc.IncludeThoughts
			tcEmpty = false
		}
		if tc.ThinkingLevel != "" {
			ntc.ThinkingLevel = tc.ThinkingLevel
			tcEmpty = false
		}
		if !tcEmpty {
			cfg.ThinkingConfig = ntc
			empty = false
		}
	}

	// ResponseModalities and ImageConfig from provider options.
	if len(opts.ResponseModalities) > 0 {
		cfg.ResponseModalities = opts.ResponseModalities
		empty = false
	}
	if opts.ImageConfig != nil {
		ic := &nativeImageConfig{}
		icEmpty := true
		if opts.ImageConfig.AspectRatio != "" {
			ic.AspectRatio = opts.ImageConfig.AspectRatio
			icEmpty = false
		}
		if opts.ImageConfig.ImageSize != "" {
			ic.ImageSize = opts.ImageConfig.ImageSize
			icEmpty = false
		}
		if !icEmpty {
			cfg.ImageConfig = ic
			empty = false
		}
	}

	// Output schema.
	if req.Output != nil {
		switch req.Output.Type {
		case "json_object", "object", "array":
			cfg.ResponseMimeType = "application/json"
			empty = false
		}
		if (req.Output.Type == "object" || req.Output.Type == "array") && req.Output.Schema != nil {
			cfg.ResponseSchema = sanitizeMap(req.Output.Schema)
			empty = false
		}
	}

	if empty {
		return nil
	}
	return cfg
}

// parseDataURI parses a data: URI and returns the MIME type and base64-encoded data.
// Returns ok=false if the URI is not a valid data URI.
func parseDataURI(uri string) (mimeType, data string, ok bool) {
	// data:[<mediatype>][;base64],<data>
	rest, found := strings.CutPrefix(uri, "data:")
	if !found {
		return "", "", false
	}
	commaIdx := strings.Index(rest, ",")
	if commaIdx < 0 {
		return "", "", false
	}
	meta := rest[:commaIdx]
	payload := rest[commaIdx+1:]

	var mime string
	isBase64 := false
	if strings.HasSuffix(meta, ";base64") {
		isBase64 = true
		mime = strings.TrimSuffix(meta, ";base64")
	} else {
		mime = meta
	}
	if mime == "" {
		mime = "application/octet-stream"
	}

	if isBase64 {
		return mime, payload, true
	}
	// Not base64-encoded: encode the raw data.
	return mime, base64.StdEncoding.EncodeToString([]byte(payload)), true
}

// guessMimeTypeFromURL returns a MIME type based on common URL file extensions.
func guessMimeTypeFromURL(url string) string {
	lower := strings.ToLower(url)
	switch {
	case strings.HasSuffix(lower, ".png"):
		return "image/png"
	case strings.HasSuffix(lower, ".jpg"), strings.HasSuffix(lower, ".jpeg"):
		return "image/jpeg"
	case strings.HasSuffix(lower, ".gif"):
		return "image/gif"
	case strings.HasSuffix(lower, ".webp"):
		return "image/webp"
	case strings.HasSuffix(lower, ".pdf"):
		return "application/pdf"
	case strings.HasSuffix(lower, ".mp4"):
		return "video/mp4"
	case strings.HasSuffix(lower, ".mp3"):
		return "audio/mpeg"
	case strings.HasSuffix(lower, ".wav"):
		return "audio/wav"
	default:
		return "application/octet-stream"
	}
}

func boolPtr(b bool) *bool {
	return &b
}
