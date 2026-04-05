package mcp

import "context"

// Transport is the interface for MCP communication channels.
// Implementations handle the underlying protocol (stdio, HTTP, etc.)
// and deliver parsed JSON-RPC messages via callbacks.
type Transport interface {
	// Start initializes the transport (e.g., spawns a process or opens a connection).
	// The context controls startup cancellation.
	Start(ctx context.Context) error

	// Send transmits a JSON-RPC message through the transport.
	Send(msg JSONRPCMessage) error

	// Close shuts down the transport and releases resources.
	Close() error

	// SetHandlers registers callbacks for incoming messages, transport closure,
	// and transport errors. Must be called before Start.
	SetHandlers(onMessage func(JSONRPCMessage), onClose func(), onError func(error))
}
