// Package uistream provides an SSE adapter that translates engine.StepEvents into
// the AI SDK UI message stream wire format consumed by second-brain-api frontends.
package uistream

// Chunk type names as used in the x-vercel-ai-ui-message-stream protocol.
const (
	ChunkStart               = "start"
	ChunkStartStep           = "start-step"
	ChunkTextStart           = "text-start"
	ChunkTextDelta           = "text-delta"
	ChunkTextEnd             = "text-end"
	ChunkReasoningStart      = "reasoning-start"
	ChunkReasoningDelta      = "reasoning-delta"
	ChunkReasoningEnd        = "reasoning-end"
	ChunkToolInputStart      = "tool-input-start"
	ChunkToolInputDelta      = "tool-input-delta"
	ChunkToolInputAvailable  = "tool-input-available"
	ChunkToolOutputAvailable = "tool-output-available"
	ChunkFinishStep          = "finish-step"
	ChunkFinish              = "finish"
	ChunkError               = "error"
)
