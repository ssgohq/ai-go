package ai

// CallSettings controls model behavior per-request.
// All pointer fields are optional; nil means "use the model default".
type CallSettings struct {
	// Temperature controls randomness. Lower values make output more deterministic.
	Temperature *float32
	// MaxTokens limits the number of tokens in the completion (0 = model default).
	MaxTokens int
	// TopP enables nucleus sampling. Set either Temperature or TopP, not both.
	TopP *float32
	// TopK limits the next-token candidates to the top K options.
	// Not supported by all providers (e.g. OpenAI ignores it).
	TopK *int
	// Seed requests deterministic output. Support varies by provider.
	Seed *int
	// StopSequences causes the model to stop when any of these strings is generated.
	StopSequences []string
}

// Usage holds token counts for a completion.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// FinishReason indicates why the model stopped generating.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonError         FinishReason = "error"
	FinishReasonUnknown       FinishReason = "unknown"
)
