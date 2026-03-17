package gemini

import (
	"encoding/json"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

func TestEncodeNativeRequest(t *testing.T) {
	t.Run("simple text-only user message", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("Hello")}},
			},
		}
		nr := encodeNativeRequest(req)

		assertNilPtr(t, nr.SystemInstruction)
		assertEqual(t, len(nr.Contents), 1)
		assertEqual(t, nr.Contents[0].Role, "user")
		assertEqual(t, len(nr.Contents[0].Parts), 1)
		assertEqual(t, nr.Contents[0].Parts[0].Text, "Hello")
	})

	t.Run("system from req.System field", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			System: "You are a helpful assistant.",
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("Hi")}},
			},
		}
		nr := encodeNativeRequest(req)

		assertNotNilPtr(t, nr.SystemInstruction)
		assertEqual(t, len(nr.SystemInstruction.Parts), 1)
		assertEqual(t, nr.SystemInstruction.Parts[0].Text, "You are a helpful assistant.")
		assertEqual(t, len(nr.Contents), 1)
		assertEqual(t, nr.Contents[0].Role, "user")
	})

	t.Run("system from leading system messages", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			System: "Base system.",
			Messages: []ai.Message{
				ai.SystemMessage("Extra system context."),
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("Hi")}},
			},
		}
		nr := encodeNativeRequest(req)

		assertNotNilPtr(t, nr.SystemInstruction)
		assertEqual(t, len(nr.SystemInstruction.Parts), 2)
		assertEqual(t, nr.SystemInstruction.Parts[0].Text, "Base system.")
		assertEqual(t, nr.SystemInstruction.Parts[1].Text, "Extra system context.")
		assertEqual(t, len(nr.Contents), 1)
		assertEqual(t, nr.Contents[0].Role, "user")
	})

	t.Run("multi-turn conversation", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("Hello")}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{ai.TextPart("Hi there!")}},
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("How are you?")}},
			},
		}
		nr := encodeNativeRequest(req)

		assertEqual(t, len(nr.Contents), 3)
		assertEqual(t, nr.Contents[0].Role, "user")
		assertEqual(t, nr.Contents[0].Parts[0].Text, "Hello")
		assertEqual(t, nr.Contents[1].Role, "model")
		assertEqual(t, nr.Contents[1].Parts[0].Text, "Hi there!")
		assertEqual(t, nr.Contents[2].Role, "user")
		assertEqual(t, nr.Contents[2].Parts[0].Text, "How are you?")
	})

	t.Run("tool call in assistant message", func(t *testing.T) {
		args := json.RawMessage(`{"query":"weather"}`)
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("What's the weather?")}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{
					{
						Type:         ai.ContentPartTypeToolCall,
						ToolCallID:   "call_1",
						ToolCallName: "get_weather",
						ToolCallArgs: args,
					},
				}},
			},
		}
		nr := encodeNativeRequest(req)

		assertEqual(t, len(nr.Contents), 2)
		modelParts := nr.Contents[1].Parts
		assertEqual(t, len(modelParts), 1)
		assertNotNilPtr(t, modelParts[0].FunctionCall)
		assertEqual(t, modelParts[0].FunctionCall.Name, "get_weather")
		assertEqual(t, string(modelParts[0].FunctionCall.Args), `{"query":"weather"}`)
	})

	t.Run("tool result message", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleTool, Content: []ai.ContentPart{
					ai.ToolResultPart("call_1", "get_weather", "Sunny, 72°F"),
				}},
			},
		}
		nr := encodeNativeRequest(req)

		assertEqual(t, len(nr.Contents), 1)
		assertEqual(t, nr.Contents[0].Role, "user")
		parts := nr.Contents[0].Parts
		assertEqual(t, len(parts), 1)
		assertNotNilPtr(t, parts[0].FunctionResponse)
		assertEqual(t, parts[0].FunctionResponse.Name, "get_weather")

		resp, ok := parts[0].FunctionResponse.Response.(map[string]string)
		assertTrue(t, ok)
		assertEqual(t, resp["name"], "get_weather")
		assertEqual(t, resp["content"], "Sunny, 72°F")
	})

	t.Run("reasoning/thought parts with ThoughtSignature", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{
					{
						Type:             ai.ContentPartTypeReasoning,
						ReasoningText:    "Let me think about this...",
						ThoughtSignature: "c2lnbmF0dXJl",
					},
					ai.TextPart("The answer is 42."),
				}},
			},
		}
		nr := encodeNativeRequest(req)

		parts := nr.Contents[0].Parts
		assertEqual(t, len(parts), 2)

		// Reasoning part.
		assertEqual(t, parts[0].Text, "Let me think about this...")
		assertNotNilPtr(t, parts[0].Thought)
		assertTrue(t, *parts[0].Thought)
		assertEqual(t, parts[0].ThoughtSignature, "c2lnbmF0dXJl")

		// Text part.
		assertEqual(t, parts[1].Text, "The answer is 42.")
		assertNilPtr(t, parts[1].Thought)
	})

	t.Run("tool call with ThoughtSignature", func(t *testing.T) {
		args := json.RawMessage(`{"x":1}`)
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{
					{
						Type:             ai.ContentPartTypeToolCall,
						ToolCallID:       "call_2",
						ToolCallName:     "compute",
						ToolCallArgs:     args,
						ThoughtSignature: "dGhvdWdodA==",
					},
				}},
			},
		}
		nr := encodeNativeRequest(req)

		parts := nr.Contents[0].Parts
		assertEqual(t, len(parts), 1)
		assertNotNilPtr(t, parts[0].FunctionCall)
		assertEqual(t, parts[0].ThoughtSignature, "dGhvdWdodA==")
	})

	t.Run("image URL as file data", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{
					{Type: ai.ContentPartTypeImageURL, ImageURL: "https://example.com/photo.jpg"},
				}},
			},
		}
		nr := encodeNativeRequest(req)

		parts := nr.Contents[0].Parts
		assertEqual(t, len(parts), 1)
		assertNotNilPtr(t, parts[0].FileData)
		assertEqual(t, parts[0].FileData.FileUri, "https://example.com/photo.jpg")
		assertEqual(t, parts[0].FileData.MimeType, "image/jpeg")
	})

	t.Run("image data URI as inline data", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{
					{Type: ai.ContentPartTypeImageURL, ImageURL: "data:image/png;base64,iVBOR"},
				}},
			},
		}
		nr := encodeNativeRequest(req)

		parts := nr.Contents[0].Parts
		assertEqual(t, len(parts), 1)
		assertNotNilPtr(t, parts[0].InlineData)
		assertEqual(t, parts[0].InlineData.MimeType, "image/png")
		assertEqual(t, parts[0].InlineData.Data, "iVBOR")
	})

	t.Run("file with inline Data bytes", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{
					ai.FileDataPart([]byte("PDF content"), "application/pdf", "doc.pdf"),
				}},
			},
		}
		nr := encodeNativeRequest(req)

		parts := nr.Contents[0].Parts
		assertEqual(t, len(parts), 1)
		assertNotNilPtr(t, parts[0].InlineData)
		assertEqual(t, parts[0].InlineData.MimeType, "application/pdf")
		assertEqual(t, parts[0].InlineData.Data, "UERGIGNvbnRlbnQ=")
	})

	t.Run("file with URL", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{
					ai.FilePart("https://storage.googleapis.com/bucket/file.pdf", "application/pdf"),
				}},
			},
		}
		nr := encodeNativeRequest(req)

		parts := nr.Contents[0].Parts
		assertEqual(t, len(parts), 1)
		assertNotNilPtr(t, parts[0].FileData)
		assertEqual(t, parts[0].FileData.FileUri, "https://storage.googleapis.com/bucket/file.pdf")
		assertEqual(t, parts[0].FileData.MimeType, "application/pdf")
	})

	t.Run("generation config mapping", func(t *testing.T) {
		temp := float32(0.7)
		topP := float32(0.9)
		topK := 40
		seed := 42
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("test")}},
			},
			Settings: ai.CallSettings{
				Temperature:   &temp,
				MaxTokens:     1024,
				TopP:          &topP,
				TopK:          &topK,
				Seed:          &seed,
				StopSequences: []string{"END", "STOP"},
			},
		}
		nr := encodeNativeRequest(req)

		assertNotNilPtr(t, nr.GenerationConfig)
		gc := nr.GenerationConfig
		assertNotNilPtr(t, gc.MaxOutputTokens)
		assertEqual(t, *gc.MaxOutputTokens, 1024)
		assertNotNilPtr(t, gc.Temperature)
		assertEqual(t, *gc.Temperature, float32(0.7))
		assertNotNilPtr(t, gc.TopP)
		assertEqual(t, *gc.TopP, float32(0.9))
		assertNotNilPtr(t, gc.TopK)
		assertEqual(t, *gc.TopK, 40)
		assertNotNilPtr(t, gc.Seed)
		assertEqual(t, *gc.Seed, 42)
		assertEqual(t, len(gc.StopSequences), 2)
		assertEqual(t, gc.StopSequences[0], "END")
	})

	t.Run("thinking config from provider options", func(t *testing.T) {
		budget := 8192
		include := true
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("think")}},
			},
			ProviderOptions: map[string]any{
				"gemini": ProviderOptions{
					ThinkingConfig: &ThinkingConfig{
						ThinkingBudget:  &budget,
						IncludeThoughts: &include,
						ThinkingLevel:   "high",
					},
				},
			},
		}
		nr := encodeNativeRequest(req)

		assertNotNilPtr(t, nr.GenerationConfig)
		assertNotNilPtr(t, nr.GenerationConfig.ThinkingConfig)
		tc := nr.GenerationConfig.ThinkingConfig
		assertNotNilPtr(t, tc.ThinkingBudget)
		assertEqual(t, *tc.ThinkingBudget, 8192)
		assertNotNilPtr(t, tc.IncludeThoughts)
		assertTrue(t, *tc.IncludeThoughts)
		assertEqual(t, tc.ThinkingLevel, "high")
	})

	t.Run("output schema json_object", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("json")}},
			},
			Output: ai.OutputJSONObject(),
		}
		nr := encodeNativeRequest(req)

		assertNotNilPtr(t, nr.GenerationConfig)
		assertEqual(t, nr.GenerationConfig.ResponseMimeType, "application/json")
		assertNilMap(t, nr.GenerationConfig.ResponseSchema)
	})

	t.Run("output schema object with schema", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
				"age":  map[string]any{"type": "integer"},
			},
			"required":             []any{"name"},
			"additionalProperties": false,
		}
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("structured")}},
			},
			Output: ai.OutputObject(schema),
		}
		nr := encodeNativeRequest(req)

		assertNotNilPtr(t, nr.GenerationConfig)
		assertEqual(t, nr.GenerationConfig.ResponseMimeType, "application/json")
		assertNotNilMap(t, nr.GenerationConfig.ResponseSchema)
		_, hasAdditionalProps := nr.GenerationConfig.ResponseSchema["additionalProperties"]
		assertTrue(t, !hasAdditionalProps)
		_, hasProps := nr.GenerationConfig.ResponseSchema["properties"]
		assertTrue(t, hasProps)
	})

	t.Run("output schema array", func(t *testing.T) {
		itemSchema := map[string]any{"type": "string"}
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("list")}},
			},
			Output: ai.OutputArray(itemSchema),
		}
		nr := encodeNativeRequest(req)

		assertNotNilPtr(t, nr.GenerationConfig)
		assertEqual(t, nr.GenerationConfig.ResponseMimeType, "application/json")
		assertNotNilMap(t, nr.GenerationConfig.ResponseSchema)
	})

	t.Run("empty messages produces empty contents", func(t *testing.T) {
		req := ai.LanguageModelRequest{}
		nr := encodeNativeRequest(req)

		assertNilPtr(t, nr.SystemInstruction)
		assertEqual(t, len(nr.Contents), 0)
		assertNilPtr(t, nr.GenerationConfig)
	})

	t.Run("no generation config when settings are zero", func(t *testing.T) {
		req := ai.LanguageModelRequest{
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("plain")}},
			},
		}
		nr := encodeNativeRequest(req)
		assertNilPtr(t, nr.GenerationConfig)
	})

	t.Run("JSON serialization roundtrip", func(t *testing.T) {
		temp := float32(0.5)
		budget := 4096
		req := ai.LanguageModelRequest{
			System: "Be helpful.",
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("Hello")}},
			},
			Settings: ai.CallSettings{
				Temperature: &temp,
				MaxTokens:   512,
			},
			ProviderOptions: map[string]any{
				"gemini": ProviderOptions{
					ThinkingConfig: &ThinkingConfig{ThinkingBudget: &budget},
				},
			},
		}
		nr := encodeNativeRequest(req)

		data, err := json.Marshal(nr)
		if err != nil {
			t.Fatalf("json.Marshal: %v", err)
		}

		var m map[string]any
		if err := json.Unmarshal(data, &m); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}

		_, hasContents := m["contents"]
		assertTrue(t, hasContents)
		_, hasSysInstr := m["systemInstruction"]
		assertTrue(t, hasSysInstr)
		_, hasGenCfg := m["generationConfig"]
		assertTrue(t, hasGenCfg)

		_, hasTools := m["tools"]
		assertTrue(t, !hasTools)
		_, hasSafety := m["safetySettings"]
		assertTrue(t, !hasSafety)
		_, hasCached := m["cachedContent"]
		assertTrue(t, !hasCached)
	})

	t.Run("full tool loop conversation", func(t *testing.T) {
		args := json.RawMessage(`{"city":"NYC"}`)
		req := ai.LanguageModelRequest{
			System: "You help with weather.",
			Messages: []ai.Message{
				{Role: ai.RoleUser, Content: []ai.ContentPart{ai.TextPart("Weather in NYC?")}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{
					{
						Type:             ai.ContentPartTypeReasoning,
						ReasoningText:    "User wants weather, I should call the tool.",
						ThoughtSignature: "dGhpbms=",
					},
					{
						Type:             ai.ContentPartTypeToolCall,
						ToolCallID:       "call_w1",
						ToolCallName:     "get_weather",
						ToolCallArgs:     args,
						ThoughtSignature: "dGhpbms=",
					},
				}},
				{Role: ai.RoleTool, Content: []ai.ContentPart{
					ai.ToolResultPart("call_w1", "get_weather", `{"temp":72,"condition":"sunny"}`),
				}},
				{Role: ai.RoleAssistant, Content: []ai.ContentPart{
					ai.TextPart("It's 72°F and sunny in NYC!"),
				}},
			},
		}
		nr := encodeNativeRequest(req)

		assertNotNilPtr(t, nr.SystemInstruction)
		assertEqual(t, len(nr.Contents), 4)

		assertEqual(t, nr.Contents[0].Role, "user")
		assertEqual(t, nr.Contents[1].Role, "model")
		assertEqual(t, len(nr.Contents[1].Parts), 2)
		assertTrue(t, *nr.Contents[1].Parts[0].Thought)
		assertNotNilPtr(t, nr.Contents[1].Parts[1].FunctionCall)
		assertEqual(t, nr.Contents[2].Role, "user")
		assertNotNilPtr(t, nr.Contents[2].Parts[0].FunctionResponse)
		assertEqual(t, nr.Contents[3].Role, "model")
		assertEqual(t, nr.Contents[3].Parts[0].Text, "It's 72°F and sunny in NYC!")
	})
}

// Test helpers

func assertEqual[T comparable](t *testing.T, got, want T) {
	t.Helper()
	if got != want {
		t.Errorf("got %v, want %v", got, want)
	}
}

func assertTrue(t *testing.T, v bool) {
	t.Helper()
	if !v {
		t.Error("expected true, got false")
	}
}

func assertNilPtr[T any](t *testing.T, v *T) {
	t.Helper()
	if v != nil {
		t.Errorf("expected nil, got %v", v)
	}
}

func assertNilMap[K comparable, V any](t *testing.T, v map[K]V) {
	t.Helper()
	if v != nil {
		t.Errorf("expected nil map, got %v", v)
	}
}

func assertNotNilPtr[T any](t *testing.T, v *T) {
	t.Helper()
	if v == nil {
		t.Error("expected non-nil, got nil")
	}
}

func assertNotNilMap[K comparable, V any](t *testing.T, v map[K]V) {
	t.Helper()
	if v == nil {
		t.Error("expected non-nil map, got nil")
	}
}
