package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// emitStructuredOutput makes a final constrained LLM call when an OutputSchema is configured.
func emitStructuredOutput(ctx context.Context, out chan<- StepEvent, params RunParams, history []Message) {
	if params.Request.Output == nil || params.Request.Output.Type == "text" {
		return
	}

	msgs := make([]Message, len(history)+1)
	copy(msgs, history)
	msgs[len(history)] = Message{
		Role:    "user",
		Content: []ContentPart{{Type: "text", Text: "Now produce the structured output as requested."}},
	}

	req := Request{
		Messages: msgs,
		Output:   params.Request.Output,
		Settings: params.Request.Settings,
	}

	eventCh, err := params.Model.Stream(ctx, req)
	if err != nil {
		out <- StepEvent{Type: StepEventError, Error: fmt.Errorf("structured output call: %w", err)}
		return
	}

	var b strings.Builder
	for ev := range eventCh {
		if ev.Type == StreamEventTextDelta {
			b.WriteString(ev.TextDelta)
		}
		if ev.Type == StreamEventError {
			return
		}
	}

	parsed := parseStructuredOutput(b.String())
	if parsed != nil {
		out <- StepEvent{Type: StepEventStructuredOutput, StructuredOutput: parsed}
	}
}

// parseStructuredOutput extracts valid JSON from content, stripping markdown fences if present.
func parseStructuredOutput(content string) json.RawMessage {
	content = trimMarkdownFence(content)
	if json.Valid([]byte(content)) {
		return json.RawMessage(content)
	}
	return nil
}

// trimMarkdownFence strips ```json ... ``` or ``` ... ``` fencing.
func trimMarkdownFence(s string) string {
	s = strings.TrimSpace(s)
	for _, prefix := range []string{"```json\n", "```json\r\n", "```\n", "```\r\n"} {
		if strings.HasPrefix(s, prefix) {
			s = s[len(prefix):]
			break
		}
	}
	for _, suffix := range []string{"\n```", "\r\n```", "```"} {
		if strings.HasSuffix(s, suffix) {
			s = s[:len(s)-len(suffix)]
			break
		}
	}
	return strings.TrimSpace(s)
}
