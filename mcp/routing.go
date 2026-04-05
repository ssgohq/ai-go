package mcp

import (
	"strings"

	"github.com/open-ai-sdk/ai-go/ai"
)

// Routing specifies MCP tool routing preferences.
// When provided, it controls which MCP servers' tools are included.
type Routing struct {
	// PreferredServers restricts tools to only these servers.
	// If empty, no preference filtering is applied.
	PreferredServers []string
	// BlockedServers excludes tools from these servers.
	BlockedServers []string
	// FallbackAllowed determines whether all tools are included
	// when PreferredServers yields zero tools. Default: true.
	FallbackAllowed *bool
	// Reason describes why this routing was applied (for logging).
	Reason string
}

// FilterToolDefs filters MCP tool definitions based on routing preferences.
// Each MCPToolDef's ServerID field is matched against the routing policy.
// Returns filtered slice. If routing is nil, returns all tools unchanged.
func FilterToolDefs(tools []MCPToolDef, routing *Routing) []MCPToolDef {
	if routing == nil {
		return tools
	}

	blocked := makeSet(routing.BlockedServers)
	preferred := makeSet(routing.PreferredServers)

	// Remove blocked servers first.
	var unblocked []MCPToolDef
	for _, t := range tools {
		if blocked[strings.ToLower(t.ServerID)] {
			continue
		}
		unblocked = append(unblocked, t)
	}

	// If no preferred servers, return unblocked set.
	if len(preferred) == 0 {
		return unblocked
	}

	// Filter to preferred servers only.
	var result []MCPToolDef
	for _, t := range unblocked {
		if preferred[strings.ToLower(t.ServerID)] {
			result = append(result, t)
		}
	}

	// Fallback: if no preferred tools found and fallback is allowed, keep all unblocked.
	if len(result) == 0 {
		if routing.FallbackAllowed == nil || *routing.FallbackAllowed {
			return unblocked
		}
		return nil
	}

	return result
}

// FilterToolSet filters an ai.ToolSet based on routing preferences.
// Uses ServerFromQualifiedName to extract the server name from each tool's
// qualified name. Returns a new ToolSet with filtered definitions and the
// same Executor. If routing is nil, returns the original ToolSet unchanged.
func FilterToolSet(ts *ai.ToolSet, routing *Routing) *ai.ToolSet {
	if routing == nil || ts == nil {
		return ts
	}

	blocked := makeSet(routing.BlockedServers)
	preferred := makeSet(routing.PreferredServers)

	// Remove blocked servers first.
	var unblocked []ai.ToolDefinition
	for _, def := range ts.Definitions {
		server := strings.ToLower(ServerFromQualifiedName(def.Name))
		if blocked[server] {
			continue
		}
		unblocked = append(unblocked, def)
	}

	// If no preferred servers, return unblocked set.
	if len(preferred) == 0 {
		return &ai.ToolSet{Definitions: unblocked, Executor: ts.Executor}
	}

	// Filter to preferred servers only.
	var result []ai.ToolDefinition
	for _, def := range unblocked {
		server := strings.ToLower(ServerFromQualifiedName(def.Name))
		if preferred[server] {
			result = append(result, def)
		}
	}

	// Fallback: if no preferred tools found and fallback is allowed, keep all unblocked.
	if len(result) == 0 {
		if routing.FallbackAllowed == nil || *routing.FallbackAllowed {
			return &ai.ToolSet{Definitions: unblocked, Executor: ts.Executor}
		}
		return &ai.ToolSet{Definitions: nil, Executor: ts.Executor}
	}

	return &ai.ToolSet{Definitions: result, Executor: ts.Executor}
}

// makeSet builds a case-insensitive lookup set from a slice of strings.
func makeSet(ss []string) map[string]bool {
	m := make(map[string]bool, len(ss))
	for _, s := range ss {
		m[strings.ToLower(s)] = true
	}
	return m
}
