package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/open-ai-sdk/ai-go/ai"
)

// MCPToolExecutor implements ai.ToolExecutor by routing tool calls to the
// correct MCP server client based on a canonical-name routing table.
type MCPToolExecutor struct {
	clients map[string]*Client // serverName -> client
	routing map[string]toolRoute
}

// toolRoute maps a canonical tool name back to the originating server and
// the tool's original (unqualified) name.
type toolRoute struct {
	serverName string
	toolName   string
}

// Execute looks up the canonical tool name in the routing table and forwards
// the call to the appropriate MCP server client.
func (e *MCPToolExecutor) Execute(ctx context.Context, canonicalName, argsJSON string) (string, error) {
	route, ok := e.routing[canonicalName]
	if !ok {
		return "", fmt.Errorf("mcp.MCPToolExecutor: unknown tool %q", canonicalName)
	}

	client, ok := e.clients[route.serverName]
	if !ok {
		return "", fmt.Errorf("mcp.MCPToolExecutor: no client for server %q", route.serverName)
	}

	var args map[string]any
	if argsJSON != "" && argsJSON != "{}" {
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			return "", fmt.Errorf("mcp.MCPToolExecutor: parse args for %q: %w", canonicalName, err)
		}
	}

	result, err := client.CallTool(ctx, route.toolName, args)
	if err != nil {
		return "", fmt.Errorf("mcp.MCPToolExecutor: call %q on server %q: %w", route.toolName, route.serverName, err)
	}

	if result.IsError {
		return "", fmt.Errorf("mcp.MCPToolExecutor: tool %q returned error: %s", canonicalName, contentToString(result.Content))
	}

	return contentToString(result.Content), nil
}

// ToolSetFromClients creates an ai.ToolSet from multiple named MCP server
// clients. Each tool is given a server-qualified canonical name using the
// format sanitize(serverName) + "_" + sanitize(toolName). The returned
// executor routes calls back to the correct server.
func ToolSetFromClients(clients map[string]*Client) (*ai.ToolSet, error) {
	executor := &MCPToolExecutor{
		clients: clients,
		routing: make(map[string]toolRoute),
	}
	var defs []ai.ToolDefinition

	for serverName, client := range clients {
		res, err := client.ListTools(context.Background())
		if err != nil {
			return nil, fmt.Errorf("mcp.ToolSetFromClients: list tools from %q: %w", serverName, err)
		}

		for _, tool := range res.Tools {
			canonical := QualifiedName(serverName, tool.Name)

			if _, exists := executor.routing[canonical]; exists {
				return nil, fmt.Errorf("mcp.ToolSetFromClients: duplicate tool name %q", canonical)
			}

			executor.routing[canonical] = toolRoute{
				serverName: serverName,
				toolName:   tool.Name,
			}

			defs = append(defs, ai.ToolDefinition{
				Name:        canonical,
				Description: tool.Description,
				InputSchema: tool.InputSchema,
			})
		}
	}

	return &ai.ToolSet{
		Definitions: defs,
		Executor:    executor,
	}, nil
}

// ToolSetFromClient creates an ai.ToolSet from a single named MCP client.
func ToolSetFromClient(serverName string, client *Client) (*ai.ToolSet, error) {
	return ToolSetFromClients(map[string]*Client{serverName: client})
}

// contentToString converts MCP content parts into a single string.
// Text parts are joined with newlines; non-text parts are noted with
// a placeholder.
func contentToString(parts []ContentPart) string {
	if len(parts) == 0 {
		return ""
	}
	if len(parts) == 1 && parts[0].Type == "text" {
		return parts[0].Text
	}

	var b strings.Builder
	for i, p := range parts {
		if i > 0 {
			b.WriteString("\n")
		}
		switch p.Type {
		case "text":
			b.WriteString(p.Text)
		case "image":
			fmt.Fprintf(&b, "[image: %s]", p.MimeType)
		case "resource":
			b.WriteString("[embedded resource]")
		default:
			fmt.Fprintf(&b, "[%s content]", p.Type)
		}
	}
	return b.String()
}


