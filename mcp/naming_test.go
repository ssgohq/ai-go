package mcp

import "testing"

func TestSanitize(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"basic passthrough", "hello_world", "hello_world"},
		{"special chars to underscore", "hello.world/v2", "hello_world_v2"},
		{"consecutive special chars", "a..b//c", "a__b__c"},
		{"all special chars", "a!b@c#d$e%f", "a_b_c_d_e_f"},
		{"empty string", "", ""},
		{"hyphens preserved", "my-server-name", "my-server-name"},
		{"alphanumeric passthrough", "abc123XYZ", "abc123XYZ"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sanitize(tt.input)
			if got != tt.want {
				t.Errorf("Sanitize(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestQualifiedName(t *testing.T) {
	tests := []struct {
		name       string
		serverName string
		toolName   string
		want       string
	}{
		{"basic format", "browsermcp", "open_tab", "browsermcp_open_tab"},
		{"special chars in both", "my-server.v2", "do-thing/now", "my-server_v2_do-thing_now"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := QualifiedName(tt.serverName, tt.toolName)
			if got != tt.want {
				t.Errorf("QualifiedName(%q, %q) = %q, want %q", tt.serverName, tt.toolName, got, tt.want)
			}
		})
	}
}

func TestCanonicalID(t *testing.T) {
	tests := []struct {
		name       string
		serverName string
		toolName   string
		want       string
	}{
		{"basic format", "browsermcp", "click", "mcp__browsermcp__click"},
		{"special chars", "my-server.v2", "do-thing/now", "mcp__my-server_v2__do-thing_now"},
		{"preserves underscores", "my_server", "my_tool", "mcp__my_server__my_tool"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CanonicalID(tt.serverName, tt.toolName)
			if got != tt.want {
				t.Errorf("CanonicalID(%q, %q) = %q, want %q", tt.serverName, tt.toolName, got, tt.want)
			}
		})
	}
}

func TestServerFromQualifiedName(t *testing.T) {
	tests := []struct {
		name          string
		qualifiedName string
		want          string
	}{
		{"basic extraction", "browsermcp_open_tab", "browsermcp"},
		{"no underscore returns empty", "notool", ""},
		{"multiple underscores returns first part", "my_server_tool", "my"},
		{"leading underscore returns empty", "_tool", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ServerFromQualifiedName(tt.qualifiedName)
			if got != tt.want {
				t.Errorf("ServerFromQualifiedName(%q) = %q, want %q", tt.qualifiedName, got, tt.want)
			}
		})
	}
}
