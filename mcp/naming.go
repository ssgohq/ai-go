package mcp

import "strings"

// Sanitize replaces characters not in [a-zA-Z0-9_-] with underscore.
// This matches OpenCode's sanitize function.
func Sanitize(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			b[i] = c
		} else {
			b[i] = '_'
		}
	}
	return string(b)
}

// QualifiedName returns a server-qualified tool name in format:
//
//	sanitize(serverName) + "_" + sanitize(toolName)
//
// This is the map key format used by OpenCode for MCP tools.
// Example: QualifiedName("browsermcp", "open_tab") → "browsermcp_open_tab"
func QualifiedName(serverName, toolName string) string {
	return Sanitize(serverName) + "_" + Sanitize(toolName)
}

// CanonicalID returns a canonical MCP tool ID in format:
//
//	"mcp__" + sanitize(serverName) + "__" + sanitize(toolName)
//
// This is the format used for display and routing policy.
// Example: CanonicalID("browsermcp", "open_tab") → "mcp__browsermcp__open_tab"
func CanonicalID(serverName, toolName string) string {
	return "mcp__" + Sanitize(serverName) + "__" + Sanitize(toolName)
}

// ServerFromQualifiedName attempts to extract the server name from a qualified
// tool name. It splits on the first "_" and returns the part before it.
// Returns empty string if the name doesn't appear to be qualified.
func ServerFromQualifiedName(qualifiedName string) string {
	i := strings.IndexByte(qualifiedName, '_')
	if i <= 0 {
		return ""
	}
	return qualifiedName[:i]
}
