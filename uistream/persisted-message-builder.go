package uistream

import (
	"encoding/json"
	"strings"
)

// PersistedMessageBuilder accumulates typed parts from stream chunks
// for persistence to a database.
type PersistedMessageBuilder struct {
	textAccum      strings.Builder
	reasoningAccum strings.Builder
	lastSignature  string

	// pendingTool tracks in-progress tool calls keyed by toolCallId
	pendingTool map[string]*toolInvocationPart

	// ordered list of finalized parts
	parts    []any
	metadata json.RawMessage

	// uiSpec accumulates data-ui-spec patches into a single final spec.
	// Each patch is either a full spec replacement or an RFC 6902 JSON Patch operation.
	uiSpec    map[string]any
	hasUISpec bool
}

type toolInvocationPart struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
	State      string `json:"state"`
	Input      any    `json:"input,omitempty"`
	Output     any    `json:"output,omitempty"`
	ErrorText  string `json:"errorText,omitempty"`
}

// NewPersistedMessageBuilder creates a new builder.
func NewPersistedMessageBuilder() *PersistedMessageBuilder {
	return &PersistedMessageBuilder{
		pendingTool: make(map[string]*toolInvocationPart),
	}
}

// ObserveChunk processes a single stream chunk, accumulating state for persistence.
func (b *PersistedMessageBuilder) ObserveChunk(c Chunk) {
	switch c.Type {
	case ChunkTextStart:
		b.observeTextStart()
	case ChunkTextDelta:
		b.observeTextDelta(c)
	case ChunkTextEnd:
		b.observeTextEnd()
	case ChunkReasoningStart:
		b.observeReasoningStart()
	case ChunkReasoningDelta:
		b.observeReasoningDelta(c)
	case ChunkReasoningEnd:
		b.observeReasoningEnd(c)
	case ChunkToolInputAvailable:
		b.observeToolInput(c)
	case ChunkToolOutputAvailable:
		b.observeToolOutput(c)
	case ChunkToolInputError:
		b.observeToolInputError(c)
	case ChunkToolOutputError:
		b.observeToolOutputError(c)
	case ChunkToolOutputDenied:
		b.observeToolOutputDenied(c)
	case ChunkSourceURL:
		b.observeSourceURL(c)
	case ChunkSourceDocument:
		b.observeSourceDocument(c)
	case ChunkStartStep:
		b.observeStartStep()
	case ChunkFile:
		b.observeFile(c)
	case ChunkMessageMetadata:
		b.observeMessageMetadata(c)
	default:
		b.observeDataChunk(c)
	}
}

func (b *PersistedMessageBuilder) observeStartStep() {
	b.parts = append(b.parts, map[string]any{"type": "step-start"})
}

func (b *PersistedMessageBuilder) observeTextStart() {
	b.textAccum.Reset()
}

func (b *PersistedMessageBuilder) observeTextDelta(c Chunk) {
	if delta, ok := c.Fields["delta"].(string); ok {
		b.textAccum.WriteString(delta)
	}
}

func (b *PersistedMessageBuilder) observeTextEnd() {
	text := b.textAccum.String()
	if text != "" {
		b.parts = append(b.parts, map[string]any{"type": "text", "text": text})
		b.textAccum.Reset()
	}
}

func (b *PersistedMessageBuilder) observeReasoningStart() {
	b.reasoningAccum.Reset()
	b.lastSignature = ""
}

func (b *PersistedMessageBuilder) observeReasoningDelta(c Chunk) {
	if delta, ok := c.Fields["delta"].(string); ok {
		b.reasoningAccum.WriteString(delta)
	}
	if sig, ok := c.Fields["signature"].(string); ok && sig != "" {
		b.lastSignature = sig
	}
}

func (b *PersistedMessageBuilder) observeMessageMetadata(c Chunk) {
	if meta := c.Fields["messageMetadata"]; meta != nil {
		if raw, err := json.Marshal(meta); err == nil {
			b.metadata = raw
		}
	}
}

func (b *PersistedMessageBuilder) observeReasoningEnd(c Chunk) {
	if sig, ok := c.Fields["signature"].(string); ok && sig != "" {
		b.lastSignature = sig
	}
	reasoning := b.reasoningAccum.String()
	if reasoning != "" {
		part := map[string]any{"type": "reasoning", "reasoning": reasoning}
		if b.lastSignature != "" {
			part["signature"] = b.lastSignature
		}
		b.parts = append(b.parts, part)
		b.reasoningAccum.Reset()
		b.lastSignature = ""
	}
}

func (b *PersistedMessageBuilder) observeToolInput(c Chunk) {
	tcID, ok1 := c.Fields["toolCallId"].(string)
	toolName, ok2 := c.Fields["toolName"].(string)
	_ = ok1
	_ = ok2
	input := c.Fields["input"]
	if tcID == "" {
		return
	}
	if _, exists := b.pendingTool[tcID]; !exists {
		b.pendingTool[tcID] = &toolInvocationPart{
			ToolCallID: tcID,
			ToolName:   toolName,
			State:      "input-available",
		}
	}
	b.pendingTool[tcID].Input = input
	b.pendingTool[tcID].ToolName = toolName
}

func (b *PersistedMessageBuilder) observeToolOutput(c Chunk) {
	tcID, ok := c.Fields["toolCallId"].(string)
	_ = ok
	output := c.Fields["output"]
	if tcID == "" {
		return
	}
	tool, exists := b.pendingTool[tcID]
	if !exists {
		tool = &toolInvocationPart{ToolCallID: tcID, State: "output-available"}
		b.pendingTool[tcID] = tool
	}
	tool.Output = output
	tool.State = "output-available"
	part := map[string]any{
		"type":       "tool-invocation",
		"toolCallId": tool.ToolCallID,
		"toolName":   tool.ToolName,
		"state":      tool.State,
		"input":      tool.Input,
		"output":     tool.Output,
	}
	applyV6ToolFields(part, c.Fields)
	b.parts = append(b.parts, part)
	delete(b.pendingTool, tcID)
}

func (b *PersistedMessageBuilder) observeToolInputError(c Chunk) {
	tcID, ok1 := c.Fields["toolCallId"].(string)
	toolName, ok2 := c.Fields["toolName"].(string)
	errText, ok3 := c.Fields["errorText"].(string)
	_ = ok1
	_ = ok2
	_ = ok3
	if tcID == "" {
		return
	}
	part := map[string]any{
		"type":       "tool-invocation",
		"toolCallId": tcID,
		"toolName":   toolName,
		"state":      "error",
		"input":      c.Fields["input"],
		"errorText":  errText,
	}
	applyV6ToolFields(part, c.Fields)
	b.parts = append(b.parts, part)
	delete(b.pendingTool, tcID)
}

func (b *PersistedMessageBuilder) observeToolOutputError(c Chunk) {
	tcID, ok1 := c.Fields["toolCallId"].(string)
	errText, ok2 := c.Fields["errorText"].(string)
	_ = ok1
	_ = ok2
	if tcID == "" {
		return
	}
	tool, exists := b.pendingTool[tcID]
	if !exists {
		tool = &toolInvocationPart{ToolCallID: tcID}
		b.pendingTool[tcID] = tool
	}
	part := map[string]any{
		"type":       "tool-invocation",
		"toolCallId": tool.ToolCallID,
		"toolName":   tool.ToolName,
		"state":      "error",
		"input":      tool.Input,
		"errorText":  errText,
	}
	applyV6ToolFields(part, c.Fields)
	b.parts = append(b.parts, part)
	delete(b.pendingTool, tcID)
}

func (b *PersistedMessageBuilder) observeToolOutputDenied(c Chunk) {
	tcID, ok := c.Fields["toolCallId"].(string)
	_ = ok
	if tcID == "" {
		return
	}
	tool, exists := b.pendingTool[tcID]
	if !exists {
		tool = &toolInvocationPart{ToolCallID: tcID}
		b.pendingTool[tcID] = tool
	}
	part := map[string]any{
		"type":       "tool-invocation",
		"toolCallId": tool.ToolCallID,
		"toolName":   tool.ToolName,
		"state":      "denied",
		"input":      tool.Input,
	}
	applyV6ToolFields(part, c.Fields)
	b.parts = append(b.parts, part)
	delete(b.pendingTool, tcID)
}

func (b *PersistedMessageBuilder) observeSourceURL(c Chunk) {
	id, ok1 := c.Fields["sourceId"].(string)
	url, ok2 := c.Fields["url"].(string)
	title, ok3 := c.Fields["title"].(string)
	_ = ok1
	_ = ok2
	_ = ok3
	b.parts = append(b.parts, map[string]any{
		"type": "source-url", "id": id, "url": url, "title": title,
	})
}

func (b *PersistedMessageBuilder) observeSourceDocument(c Chunk) {
	id, ok1 := c.Fields["sourceId"].(string)
	title, ok2 := c.Fields["title"].(string)
	mediaType, ok3 := c.Fields["mediaType"].(string)
	_ = ok1
	_ = ok2
	_ = ok3
	part := map[string]any{
		"type": "source-document", "id": id, "title": title, "mediaType": mediaType,
	}
	if fn, ok := c.Fields["filename"].(string); ok && fn != "" {
		part["filename"] = fn
	}
	if data := c.Fields["data"]; data != nil {
		part["data"] = data
	}
	if pm := c.Fields["providerMetadata"]; pm != nil {
		part["providerMetadata"] = pm
	}
	b.parts = append(b.parts, part)
}

func (b *PersistedMessageBuilder) observeFile(c Chunk) {
	part := map[string]any{"type": "file"}
	for _, key := range []string{"url", "mediaType", "name", "id", "fileId"} {
		if v, ok := c.Fields[key].(string); ok && v != "" {
			part[key] = v
		}
	}
	if data := c.Fields["data"]; data != nil {
		part["data"] = data
	}
	if pm := c.Fields["providerMetadata"]; pm != nil {
		part["providerMetadata"] = pm
	}
	b.parts = append(b.parts, part)
}

// copyOptionalBool copies a bool field from src to dst if present.
func copyOptionalBool(dst, src map[string]any, key string) {
	if v, ok := src[key].(bool); ok {
		dst[key] = v
	}
}

// copyOptionalString copies a non-empty string field from src to dst if present.
func copyOptionalString(dst, src map[string]any, key string) {
	if v, ok := src[key].(string); ok && v != "" {
		dst[key] = v
	}
}

// applyV6ToolFields copies optional v6 bool/string fields from a chunk into a part map.
func applyV6ToolFields(part, fields map[string]any) {
	copyOptionalBool(part, fields, "providerExecuted")
	copyOptionalBool(part, fields, "dynamic")
	copyOptionalBool(part, fields, "preliminary")
	copyOptionalString(part, fields, "title")
}

func (b *PersistedMessageBuilder) observeDataChunk(c Chunk) {
	// data-* chunks (non-transient); transient-data-* are excluded from parts.
	if !strings.HasPrefix(c.Type, "data-") || strings.HasPrefix(c.Type, "transient-data-") {
		return
	}
	transient, ok := c.Fields["transient"].(bool)
	_ = ok
	if transient {
		return
	}
	name := strings.TrimPrefix(c.Type, "data-")

	// data-ui-spec chunks are accumulated into a single ui-spec part so that
	// the client can decode them as UIRenderSpecPart (type="ui-spec") on history load.
	if name == "ui-spec" {
		if patch, ok := c.Fields["data"].(map[string]any); ok {
			b.applyUISpecPatch(patch)
		}
		return
	}

	b.parts = append(b.parts, map[string]any{
		"type": "data", "name": name, "data": c.Fields["data"], "isTransient": false,
	})
}

// applyUISpecPatch applies a single patch to the accumulated ui-spec.
// The patch is either an RFC 6902 JSON Patch operation (has "op" key)
// or a full spec replacement (has "root"/"elements" keys).
func (b *PersistedMessageBuilder) applyUISpecPatch(patch map[string]any) {
	if b.uiSpec == nil {
		b.uiSpec = map[string]any{}
	}
	b.hasUISpec = true
	if _, hasOp := patch["op"].(string); hasOp {
		uiSpecJSONPatch(b.uiSpec, patch)
	} else {
		// Full spec replacement.
		b.uiSpec = patch
	}
}

// uiSpecJSONPatch applies a single RFC 6902 operation to a map[string]any document.
// Supports "add", "replace", and "remove" operations.
func uiSpecJSONPatch(doc map[string]any, patch map[string]any) {
	op, _ := patch["op"].(string)
	path, _ := patch["path"].(string)
	value := patch["value"]

	// Trim leading "/" and split into path segments.
	segments := strings.Split(strings.TrimPrefix(path, "/"), "/")
	if len(segments) == 0 || (len(segments) == 1 && segments[0] == "") {
		return
	}
	switch op {
	case "add", "replace":
		uiSpecSetPath(doc, segments, value)
	case "remove":
		uiSpecRemovePath(doc, segments)
	}
}

func uiSpecSetPath(node map[string]any, segments []string, value any) {
	if len(segments) == 1 {
		node[segments[0]] = value
		return
	}
	child, ok := node[segments[0]].(map[string]any)
	if !ok {
		child = map[string]any{}
		node[segments[0]] = child
	}
	uiSpecSetPath(child, segments[1:], value)
}

func uiSpecRemovePath(node map[string]any, segments []string) {
	if len(segments) == 1 {
		delete(node, segments[0])
		return
	}
	if child, ok := node[segments[0]].(map[string]any); ok {
		uiSpecRemovePath(child, segments[1:])
	}
}

// Content returns the denormalized text content (all text parts joined).
func (b *PersistedMessageBuilder) Content() string {
	var sb strings.Builder
	for _, p := range b.parts {
		m, ok := p.(map[string]any)
		if !ok {
			continue
		}
		if m["type"] == "text" {
			if t, ok := m["text"].(string); ok {
				sb.WriteString(t)
			}
		}
	}
	return sb.String()
}

// Parts returns the serialized JSON array of typed parts.
// If any data-ui-spec chunks were observed, a single "ui-spec" part with the
// accumulated final spec is appended so that clients can decode it as UIRenderSpecPart.
func (b *PersistedMessageBuilder) Parts() json.RawMessage {
	parts := b.parts
	if b.hasUISpec {
		parts = append(parts, map[string]any{
			"type":     "ui-spec",
			"rawValue": b.uiSpec,
		})
	}
	if len(parts) == 0 {
		return nil
	}
	raw, err := json.Marshal(parts)
	if err != nil {
		return nil
	}
	return raw
}

// Metadata returns the message metadata as raw JSON (may be nil).
func (b *PersistedMessageBuilder) Metadata() json.RawMessage {
	return b.metadata
}

// MergeWithPersistence returns a MergeOption that feeds every chunk through
// the builder's ObserveChunk method during MergeStreamResult.
func MergeWithPersistence(b *PersistedMessageBuilder) MergeOption {
	return func(c *mergeConfig) {
		c.persistenceBuilder = b
	}
}
