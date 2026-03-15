# ai-go

Go SDK for building AI-powered applications with a unified provider interface.

## Features

- **Multi-provider support** — Gemini, OpenAI (Responses + Chat Completions), and any OpenAI-compatible endpoint
- **Streaming and non-streaming** — `GenerateText` and `StreamText` with automatic tool-loop orchestration
- **Tool use** — Define tools, execute them, and let the engine handle multi-step reasoning
- **Structured output** — JSON schema-constrained generation with automatic validation
- **Provider extensions** — Generic `ProviderOptions` / `ProviderMetadata` / `Warnings` without polluting core types
- **UI streaming** — SSE adapter emitting AI SDK UI message stream protocol chunks
- **Embeddings** — Unified embedding model interface

## Packages

| Package | Description |
|---------|-------------|
| `ai` | Public API: `GenerateText`, `StreamText`, request/result types, model interfaces |
| `provider/gemini` | Google Gemini provider |
| `provider/openai` | OpenAI Responses API + Chat Completions provider |
| `provider/openai_compatible` | Generic provider for any OpenAI-compatible endpoint |
| `uistream` | SSE adapter and Writer for AI SDK UI message stream protocol |
| `schema` | JSON Schema builder for structured output |

## Quick Start

```go
import (
    "github.com/ssgohq/ai-go/ai"
    "github.com/ssgohq/ai-go/provider/openai"
)

// OpenAI Chat Completions
model := openai.NewChatLanguageModel("gpt-4o", openai.Config{
    APIKey: os.Getenv("OPENAI_API_KEY"),
})

result, err := ai.GenerateText(ctx, ai.GenerateTextRequest{
    Model:    model,
    System:   "You are a helpful assistant.",
    Messages: []ai.Message{ai.UserMessage("Hello!")},
})
fmt.Println(result.Text)
```

### OpenAI Responses API

```go
model := openai.NewLanguageModel("gpt-4o", openai.Config{
    APIKey: os.Getenv("OPENAI_API_KEY"),
})

result, _ := ai.GenerateText(ctx, ai.GenerateTextRequest{
    Model:    model,
    Messages: []ai.Message{ai.UserMessage("Search the web for Go 1.25 features")},
    ProviderOptions: map[string]any{
        "openai": openai.ProviderOptions{
            EnableWebSearch: true,
            IncludeSources:  true,
        },
    },
})
```

### Gemini

```go
import "github.com/ssgohq/ai-go/provider/gemini"

model := gemini.NewLanguageModel("gemini-2.0-flash", gemini.Config{
    APIKey: os.Getenv("GEMINI_API_KEY"),
})
```

### OpenAI-Compatible Provider

```go
import "github.com/ssgohq/ai-go/provider/openai_compatible"

model := openai_compatible.NewLanguageModel("deepseek-chat", openai_compatible.Config{
    BaseURL: "https://api.deepseek.com/v1",
    APIKey:  os.Getenv("DEEPSEEK_API_KEY"),
})
```

### Streaming with UI Protocol

```go
stream := ai.StreamText(ctx, req)

var buf bytes.Buffer
adapter := uistream.NewAdapter("msg-1")
text := adapter.Stream(stream, &buf)
// buf contains SSE-encoded AI SDK UI message stream chunks
```

## Requirements

- Go 1.25+

## License

MIT — see [LICENSE](LICENSE) for details.
