package mcp

// SupportedProtocolVersions lists all MCP protocol versions this client accepts.
var SupportedProtocolVersions = []string{
	LatestProtocolVersion,
	"2025-06-18",
	"2025-03-26",
	"2024-11-05",
}

// Implementation identifies an MCP client or server.
type Implementation struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ServerCapabilities describes what the server supports.
type ServerCapabilities struct {
	Tools     *ToolsCapability     `json:"tools,omitempty"`
	Resources *ResourcesCapability `json:"resources,omitempty"`
	Prompts   *PromptsCapability   `json:"prompts,omitempty"`
}

// ToolsCapability indicates server support for tools.
type ToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// ResourcesCapability indicates server support for resources.
type ResourcesCapability struct {
	Subscribe   bool `json:"subscribe,omitempty"`
	ListChanged bool `json:"listChanged,omitempty"`
}

// PromptsCapability indicates server support for prompts.
type PromptsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// InitializeResult is the server's response to an initialize request.
type InitializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	ServerInfo      Implementation     `json:"serverInfo"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	Instructions    string             `json:"instructions,omitempty"`
}

// MCPToolDef is a tool definition returned by a server's tools/list response.
type MCPToolDef struct {
	Name         string         `json:"name"`
	Title        string         `json:"title,omitempty"`
	Description  string         `json:"description,omitempty"`
	InputSchema  map[string]any `json:"inputSchema"`
	OutputSchema map[string]any `json:"outputSchema,omitempty"`
	Annotations  map[string]any `json:"annotations,omitempty"`
	// ServerID identifies which MCP server this tool came from.
	// Set by the caller when aggregating tools from multiple servers.
	ServerID string `json:"serverID,omitempty"`
}

// ListToolsResult is the server's response to a tools/list request.
type ListToolsResult struct {
	Tools      []MCPToolDef `json:"tools"`
	NextCursor string       `json:"nextCursor,omitempty"`
}

// CallToolResult is the server's response to a tools/call request.
type CallToolResult struct {
	Content []ContentPart `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// ContentPart is a single piece of content within a tool result.
type ContentPart struct {
	Type     string `json:"type"`               // "text", "image", "resource"
	Text     string `json:"text,omitempty"`     // for type "text"
	Data     string `json:"data,omitempty"`     // for type "image" (base64)
	MimeType string `json:"mimeType,omitempty"` // for type "image"
}
