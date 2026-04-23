package ai

// Response mirrors the AI SDK Node `response` surface that callers use to
// continue multi-step conversations.
type Response struct {
	Messages []Message
}

// ResponseMessagesForStep converts a completed step into continuation messages.
func ResponseMessagesForStep(step StepOutput, tools *ToolSet) []Message {
	var messages []Message

	assistantParts := make([]ContentPart, 0, len(step.ToolCalls)+2)
	if step.Reasoning != "" {
		assistantParts = append(assistantParts, ReasoningPart(step.Reasoning))
	}
	if step.Text != "" {
		assistantParts = append(assistantParts, TextPart(step.Text))
	}
	for _, tc := range step.ToolCalls {
		assistantParts = append(assistantParts, ContentPart{
			Type:             ContentPartTypeToolCall,
			ToolCallID:       tc.ID,
			ToolCallName:     tc.Name,
			ToolCallArgs:     tc.Args,
			ThoughtSignature: tc.ThoughtSignature,
		})
	}
	if len(assistantParts) > 0 {
		messages = append(messages, Message{
			Role:    RoleAssistant,
			Content: assistantParts,
		})
	}

	for _, tr := range step.ToolResults {
		messages = append(messages, Message{
			Role: RoleTool,
			Content: []ContentPart{
				ToolResultPart(tr.ID, tr.Name, responseMessageToolOutput(tr, tools)),
			},
		})
	}

	return messages
}

// ResponseMessagesForSteps converts all completed steps into continuation
// messages in execution order.
func ResponseMessagesForSteps(steps []StepOutput, tools *ToolSet) []Message {
	if len(steps) == 0 {
		return nil
	}
	var messages []Message
	for _, step := range steps {
		messages = append(messages, ResponseMessagesForStep(step, tools)...)
	}
	return messages
}

func responseMessageToolOutput(result ToolResult, tools *ToolSet) string {
	if tools == nil {
		return result.Output
	}
	for _, def := range tools.Definitions {
		if def.Name == result.Name {
			if def.ToModelOutput != nil {
				return def.ToModelOutput(result.Output)
			}
			break
		}
	}
	return result.Output
}
