package ai

// PruneMode controls how message content is pruned.
type PruneMode string

const (
	// PruneModeNone keeps all content (default for Reasoning and ToolCalls).
	PruneModeNone PruneMode = "none"
	// PruneModeAll removes all matching content parts.
	PruneModeAll PruneMode = "all"
	// PruneModeBeforeLastMsg keeps matching parts only in the last message.
	PruneModeBeforeLastMsg PruneMode = "before-last-message"
	// PruneModeBeforeLastNMsgs keeps matching parts only in the last N messages.
	PruneModeBeforeLastNMsgs PruneMode = "before-last-n-messages"
	// PruneModeRemove drops empty messages after pruning (for EmptyMessages).
	PruneModeRemove PruneMode = "remove"
	// PruneModeKeep retains empty messages after pruning (default for EmptyMessages).
	PruneModeKeep PruneMode = "keep"
)

// PruneOptions configures which parts of messages to remove.
type PruneOptions struct {
	// Reasoning controls pruning of reasoning content parts.
	// "none" (default) = keep all, "all" = remove all,
	// "before-last-message" = keep only in last message.
	Reasoning PruneMode
	// ToolCalls controls pruning of tool_call and tool_result content parts
	// and tool-role messages.
	// "none" (default) = keep all, "all" = remove all,
	// "before-last-n-messages" = keep in last N.
	ToolCalls PruneMode
	// EmptyMessages controls handling of messages that become empty after pruning.
	// "keep" (default) = leave them, "remove" = drop them.
	EmptyMessages PruneMode
	// N is used with PruneModeBeforeLastNMsgs to specify how many recent
	// messages to preserve.
	N int
}

// PruneMessages returns a new slice of messages with content pruned according
// to opts. The original messages slice is not modified.
func PruneMessages(messages []Message, opts PruneOptions) []Message {
	if len(messages) == 0 {
		return nil
	}

	reasoning := opts.Reasoning
	if reasoning == "" {
		reasoning = PruneModeNone
	}
	toolCalls := opts.ToolCalls
	if toolCalls == "" {
		toolCalls = PruneModeNone
	}
	emptyMsgs := opts.EmptyMessages
	if emptyMsgs == "" {
		emptyMsgs = PruneModeKeep
	}

	// Fast path: nothing to prune.
	if reasoning == PruneModeNone && toolCalls == PruneModeNone {
		out := make([]Message, len(messages))
		copy(out, messages)
		return out
	}

	result := make([]Message, 0, len(messages))
	for i, msg := range messages {
		pruneR := shouldPrune(reasoning, i, len(messages), opts.N)
		pruneT := shouldPrune(toolCalls, i, len(messages), opts.N)
		if m, keep := pruneMessage(msg, pruneR, pruneT, emptyMsgs); keep {
			result = append(result, m)
		}
	}

	return result
}

// pruneMessage applies per-message pruning and returns the pruned message and
// whether it should be kept (false means drop it entirely).
func pruneMessage(msg Message, pruneReasoning, pruneToolCalls bool, emptyMsgs PruneMode) (Message, bool) {
	// Tool-role messages are entirely tool-related; prune the whole message.
	if pruneToolCalls && msg.Role == RoleTool {
		if emptyMsgs == PruneModeRemove {
			return Message{}, false
		}
		return Message{Role: msg.Role}, true
	}

	if !pruneReasoning && !pruneToolCalls {
		return msg, true
	}

	var filtered []ContentPart
	for _, part := range msg.Content {
		if pruneReasoning && part.Type == ContentPartTypeReasoning {
			continue
		}
		if pruneToolCalls &&
			(part.Type == ContentPartTypeToolCall || part.Type == ContentPartTypeToolResult) {
			continue
		}
		filtered = append(filtered, part)
	}

	if len(filtered) == 0 && emptyMsgs == PruneModeRemove {
		return Message{}, false
	}

	return Message{Role: msg.Role, Content: filtered}, true
}

// shouldPrune returns true if content at index i should be pruned given the mode.
func shouldPrune(mode PruneMode, i, total, n int) bool {
	switch mode {
	case PruneModeAll:
		return true
	case PruneModeBeforeLastMsg:
		return i < total-1
	case PruneModeBeforeLastNMsgs:
		if n <= 0 {
			n = 1
		}
		return i < total-n
	default:
		return false
	}
}
