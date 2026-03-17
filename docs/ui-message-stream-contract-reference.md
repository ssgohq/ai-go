# UI Message Stream Contract Reference

Defines the wire contract between ai-go backends and AI SDK Node v6 / Swift clients.

## Wire Format

Chunks are SSE events:

```
data: {"type":"text-delta","id":"text_1","delta":"Hello"}\n\n
data: [DONE]\n\n
```

Each line is `data: <json>\n\n`. The stream ends with `data: [DONE]\n\n` (emitted automatically after the `finish` chunk).

---

## Chunk Type Parity Table

| AI SDK Node v6 type       | ai-go constant              | Notes                                               |
|---------------------------|-----------------------------|-----------------------------------------------------|
| `start`                   | `ChunkStart`                | Carries `messageId`                                 |
| `start-step`              | `ChunkStartStep`            |                                                     |
| `text-start`              | `ChunkTextStart`            | Carries `id` (block ID)                             |
| `text-delta`              | `ChunkTextDelta`            | Carries `id`, `delta`, optional `providerMetadata`  |
| `text-end`                | `ChunkTextEnd`              | Carries `id`                                        |
| `reasoning-start`         | `ChunkReasoningStart`       | Carries `id`                                        |
| `reasoning-delta`         | `ChunkReasoningDelta`       | Carries `id`, `delta`, optional `providerMetadata`  |
| `reasoning-end`           | `ChunkReasoningEnd`         | See **Intentional Differences** for `signature`     |
| `tool-input-start`        | `ChunkToolInputStart`       | Carries `toolCallId`, `toolName`; optional `providerExecuted`, `dynamic`, `title` |
| `tool-input-delta`        | `ChunkToolInputDelta`       | Carries `toolCallId`, `inputTextDelta`; optional `providerExecuted`, `dynamic` |
| `tool-input-available`    | `ChunkToolInputAvailable`   | Carries `toolCallId`, `toolName`, `input`; optional `providerExecuted`, `dynamic`, `title` |
| `tool-output-available`   | `ChunkToolOutputAvailable`  | Carries `toolCallId`, `output`; optional `providerMetadata`, `providerExecuted`, `dynamic`, `preliminary` |
| `tool-input-error`        | `ChunkToolInputError`       | Carries `toolCallId`, `toolName`, `input`, `errorText`; optional `providerExecuted`, `dynamic`, `title` |
| `tool-output-error`       | `ChunkToolOutputError`      | Carries `toolCallId`, `errorText`; optional `providerExecuted`, `dynamic` |
| `tool-output-denied`      | `ChunkToolOutputDenied`     | Carries `toolCallId`; optional `providerExecuted`, `dynamic` |
| `tool-approval-request`   | `ChunkToolApprovalRequest`  | Carries `approvalId`, `toolCallId`, `toolName`, `args` |
| `finish-step`             | `ChunkFinishStep`           |                                                     |
| `finish`                  | `ChunkFinish`               | Carries optional `finishReason`, `messageMetadata`  |
| `error`                   | `ChunkError`                | Carries `errorText`                                 |
| `source`                  | `ChunkSource`               | Carries `id`, `url`, `title`                        |
| `sources`                 | `ChunkSources`              | Carries `sources` array                             |
| `source-url`              | `ChunkSourceURL`            | Carries `sourceId`, `url`, `title`                  |
| `source-document`         | `ChunkSourceDocument`       | Carries `sourceId`, `mediaType`, `title`; optional `filename`, `data`, `providerMetadata` |
| `message-metadata`        | `ChunkMessageMetadata`      | Carries `messageMetadata`                           |
| `abort`                   | `ChunkAbort`                | Carries optional `reason`                           |
| `file`                    | `ChunkFile`                 | Carries `url`, `mediaType`; optional `id`, `fileId`, `data`, `name`, `providerMetadata` |
| `data-<name>`             | (dynamic via `WriteData`)   | Carries `data` field; use `WriteData(name, payload)` |

---

## Chat Request Envelope

```json
{
  "id": "session-abc",
  "trigger": "submit-message",
  "messageId": "",
  "messages": [
    {
      "id": "msg-1",
      "role": "user",
      "content": "Hello",
      "metadata": {}
    }
  ],
  "body": {
    "modelId": "openai:gpt-4o",
    "system": "You are helpful.",
    "maxSteps": 5,
    "maxTokens": 2048
  },
  "metadata": {}
}
```

| Field              | Type             | Description                                                    |
|--------------------|------------------|----------------------------------------------------------------|
| `id`               | string           | Session/thread identifier                                      |
| `trigger`          | string           | `"submit-message"` (default) or `"regenerate-message"`        |
| `messageId`        | string           | Regeneration target message ID (used with `trigger=regenerate`)|
| `messages`         | EnvelopeMessage[]| Full conversation history                                      |
| `messages[].id`    | string           | Per-message ID (used for continuation)                         |
| `messages[].metadata` | object        | Per-message metadata (not forwarded to model)                  |
| `body`             | object           | Route/model hints: `modelId`, `system`, `maxSteps`, `maxTokens`|
| `metadata`         | object           | Session-level metadata for logging/tracing (not sent to model) |

### Message Part Types

- `text` — `{ type, text }`
- `image` — `{ type, url?, data?, fileId?, mediaType }`
- `file` — `{ type, url?, data?, fileId?, name?, mediaType }`
- `tool-invocation` — `{ type, toolCallId, toolName, input, output?, state }` (for history replay)
  - `state` values: `"partial-call"`, `"call"`, `"result"`, `"error"`
  - `state=result` produces both a `ToolCallPart` and a `ToolResultPart` when converted via `ToAIContentParts`

---

## Intentional Differences from AI SDK Node v6

| Topic | Node v6 | ai-go | Reason |
|-------|---------|-------|--------|
| `signature` on reasoning-end | Nested under `providerMetadata.anthropic.thinking.signature` | Top-level `signature` field on `reasoning-end` chunk | Backward compat; simplifies Go consumers |
| Message ID generation | `generateMessageId` callback option | Explicit `msgID` parameter to `ToUIMessageStream` | Simpler API; callers control ID lifecycle |
| `originalMessages` | Present in Node stream result | Not emitted | Go callers manage message history explicitly |
| `SendStart`/`SendFinish` default | `true` | `nil` (= `true`) via `*bool` | Nil pointer means "use default"; avoids forced bool in every call site |

---

## Integration Examples

### Basic stream to HTTP response

```go
func chatHandler(w http.ResponseWriter, r *http.Request) {
    var env uistream.ChatRequestEnvelope
    json.NewDecoder(r.Body).Decode(&env)

    model := openai.New("gpt-4o")
    sr := ai.StreamText(r.Context(), ai.StreamTextRequest{
        Model:    model,
        Messages: uistream.ToAIMessages(env.Messages),
    })

    msgID := uistream.ResolveMessageIDFromEnvelope(env, uuid.New())
    w.Header().Set("Content-Type", "text/event-stream")
    uistream.StreamToWriter(w, sr, msgID, uistream.ToUIStreamOptions{
        SendReasoning: true,
        SendSources:   true,
    })
}
```

### Suppress lifecycle chunks when merging streams

```go
// Outer stream manages start/finish; inner streams skip them.
innerOpts := uistream.ToUIStreamOptions{
    SendStart:  uistream.BoolPtr(false),
    SendFinish: uistream.BoolPtr(false),
}
```

### Persist typed parts with PersistedMessageBuilder

```go
builder := uistream.NewPersistedMessageBuilder()
for chunk := range chunkCh {
    builder.ObserveChunk(chunk)
}
parts := builder.Parts()   // json.RawMessage typed parts array
content := builder.Content() // plain text for full-text search
```

### Emit tool error chunks manually

```go
wr := uistream.NewWriter(w)
wr.WriteToolOutputError("tc-123", "connection timeout", nil)
// with v6 optional fields:
provExec := true
wr.WriteToolOutputError("tc-124", "timeout", &uistream.ToolChunkOpts{ProviderExecuted: &provExec})
// or for human-in-the-loop:
wr.WriteToolApprovalRequest("approval-1", "tc-456", "delete_file", args)
```
