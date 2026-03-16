package ai

import (
	"context"
	"fmt"
)

// Tool is a self-contained tool definition with its own typed executor.
// Use NewToolSet to combine multiple Tool values into a ToolSet.
type Tool struct {
	Name        string
	Description string
	InputSchema map[string]any
	Execute     func(ctx context.Context, argsJSON string) (string, error)
}

// NewToolSet creates a ToolSet from individual Tool values.
// Each Tool carries its own executor; the returned ToolSet dispatches by name.
func NewToolSet(tools ...Tool) *ToolSet {
	defs := make([]ToolDefinition, len(tools))
	dispatcher := &toolSetDispatcher{
		executors: make(map[string]func(ctx context.Context, argsJSON string) (string, error), len(tools)),
	}
	for i, t := range tools {
		defs[i] = ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
		dispatcher.executors[t.Name] = t.Execute
	}
	return &ToolSet{
		Definitions: defs,
		Executor:    dispatcher,
	}
}

// toolSetDispatcher routes tool execution to the registered function by name.
type toolSetDispatcher struct {
	executors map[string]func(ctx context.Context, argsJSON string) (string, error)
}

func (d *toolSetDispatcher) Execute(ctx context.Context, name, argsJSON string) (string, error) {
	fn, ok := d.executors[name]
	if !ok {
		return "", fmt.Errorf("unknown tool: %s", name)
	}
	return fn(ctx, argsJSON)
}
