package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
)

// ClientConfig configures an MCP client.
type ClientConfig struct {
	// Transport is the underlying communication channel.
	Transport Transport
	// Name is the client name advertised during initialization.
	// Defaults to "ai-go-mcp-client".
	Name string
	// Version is the client version advertised during initialization.
	// Defaults to "1.0.0".
	Version string
}

// Client is an MCP protocol client that communicates with a single server
// over a Transport. It handles the initialize handshake and provides
// methods for listing and calling tools.
type Client struct {
	transport    Transport
	clientInfo   Implementation
	capabilities ServerCapabilities

	mu      sync.Mutex
	pending map[int64]chan result
	nextID  atomic.Int64
	closed  bool
}

// result carries either a successful JSON-RPC response or an error.
type result struct {
	response *JSONRPCResponse
	err      error
}

// NewClient creates a new MCP client. Call Initialize to perform the
// protocol handshake before calling other methods.
func NewClient(cfg ClientConfig) *Client {
	name := cfg.Name
	if name == "" {
		name = "ai-go-mcp-client"
	}
	version := cfg.Version
	if version == "" {
		version = "1.0.0"
	}

	c := &Client{
		transport:  cfg.Transport,
		clientInfo: Implementation{Name: name, Version: version},
		pending:    make(map[int64]chan result),
	}

	cfg.Transport.SetHandlers(c.onMessage, c.onClose, c.onError)
	return c
}

// Initialize performs the MCP protocol handshake: starts the transport,
// sends an initialize request, validates the server's protocol version,
// and sends the initialized notification.
func (c *Client) Initialize(ctx context.Context) (*InitializeResult, error) {
	if err := c.transport.Start(ctx); err != nil {
		return nil, fmt.Errorf("mcp.Client.Initialize: start transport: %w", err)
	}

	params, err := json.Marshal(map[string]any{
		"protocolVersion": LatestProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo": map[string]any{
			"name":    c.clientInfo.Name,
			"version": c.clientInfo.Version,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mcp.Client.Initialize: marshal params: %w", err)
	}

	raw, err := c.request(ctx, "initialize", params)
	if err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("mcp.Client.Initialize: %w", err)
	}

	var res InitializeResult
	if err := json.Unmarshal(raw, &res); err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("mcp.Client.Initialize: parse result: %w", err)
	}

	if !isSupportedVersion(res.ProtocolVersion) {
		_ = c.Close()
		return nil, fmt.Errorf("mcp.Client.Initialize: unsupported protocol version %q", res.ProtocolVersion)
	}

	c.mu.Lock()
	c.capabilities = res.Capabilities
	c.mu.Unlock()

	// Complete the handshake with initialized notification.
	notif := NewNotification("notifications/initialized", nil)
	if err := c.transport.Send(notif); err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("mcp.Client.Initialize: send initialized notification: %w", err)
	}

	return &res, nil
}

// ListTools retrieves the list of tools available on the server.
func (c *Client) ListTools(ctx context.Context) (*ListToolsResult, error) {
	raw, err := c.request(ctx, "tools/list", nil)
	if err != nil {
		return nil, fmt.Errorf("mcp.Client.ListTools: %w", err)
	}

	var res ListToolsResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, fmt.Errorf("mcp.Client.ListTools: parse result: %w", err)
	}
	return &res, nil
}

// CallTool invokes a named tool on the server with the given arguments.
func (c *Client) CallTool(ctx context.Context, name string, args map[string]any) (*CallToolResult, error) {
	params, err := json.Marshal(map[string]any{
		"name":      name,
		"arguments": args,
	})
	if err != nil {
		return nil, fmt.Errorf("mcp.Client.CallTool: marshal params: %w", err)
	}

	raw, err := c.request(ctx, "tools/call", params)
	if err != nil {
		return nil, fmt.Errorf("mcp.Client.CallTool: %w", err)
	}

	var res CallToolResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, fmt.Errorf("mcp.Client.CallTool: parse result: %w", err)
	}
	return &res, nil
}

// Close shuts down the client and its transport.
func (c *Client) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	// Fail all pending requests.
	for id, ch := range c.pending {
		ch <- result{err: fmt.Errorf("client closed")}
		delete(c.pending, id)
	}
	c.mu.Unlock()

	return c.transport.Close()
}

// request sends a JSON-RPC request and waits for the matching response.
func (c *Client) request(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, fmt.Errorf("client closed")
	}
	c.mu.Unlock()

	id := c.nextID.Add(1)
	ch := make(chan result, 1)

	c.mu.Lock()
	c.pending[id] = ch
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}()

	msg := NewRequest(IntID(id), method, params)
	if err := c.transport.Send(msg); err != nil {
		return nil, fmt.Errorf("send %s: %w", method, err)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-ch:
		if r.err != nil {
			return nil, r.err
		}
		return r.response.Result, nil
	}
}

// onMessage dispatches incoming JSON-RPC messages to pending request handlers.
func (c *Client) onMessage(msg JSONRPCMessage) {
	switch {
	case msg.IsResponse():
		c.mu.Lock()
		ch, ok := c.pending[msg.Response.ID.num]
		c.mu.Unlock()
		if ok {
			ch <- result{response: msg.Response}
		}
	case msg.IsError():
		c.mu.Lock()
		ch, ok := c.pending[msg.Error.ID.num]
		c.mu.Unlock()
		if ok {
			ch <- result{err: fmt.Errorf("server error %d: %s", msg.Error.Error.Code, msg.Error.Error.Message)}
		}
	}
}

// onClose is called when the transport closes.
func (c *Client) onClose() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	for id, ch := range c.pending {
		ch <- result{err: fmt.Errorf("transport closed")}
		delete(c.pending, id)
	}
}

// onError is called when the transport encounters an error.
func (c *Client) onError(err error) {
	// Transport errors are non-fatal; individual request errors are
	// delivered via onMessage (JSONRPCError). We could log here if
	// the client had a logger, but for now we silently discard.
	_ = err
}

// isSupportedVersion checks whether the given protocol version is supported.
func isSupportedVersion(v string) bool {
	for _, sv := range SupportedProtocolVersions {
		if sv == v {
			return true
		}
	}
	return false
}
