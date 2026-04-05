package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
)

// LatestProtocolVersion is the MCP protocol version advertised in HTTP headers.
const LatestProtocolVersion = "2025-11-25"

// HTTPTransportConfig configures an HTTP-based MCP transport (Streamable HTTP).
type HTTPTransportConfig struct {
	// URL is the MCP server endpoint.
	URL string
	// Headers are additional HTTP headers sent with every request.
	Headers map[string]string
	// HTTPClient is an optional HTTP client; http.DefaultClient is used if nil.
	HTTPClient *http.Client
}

// HTTPTransport implements Transport using the MCP Streamable HTTP protocol.
// It POSTs JSON-RPC messages, reads responses as JSON or SSE, and optionally
// maintains an inbound SSE connection for server-initiated messages.
type HTTPTransport struct {
	config HTTPTransportConfig
	client *http.Client

	mu        sync.Mutex
	sessionID string
	cancel    context.CancelFunc
	started   bool

	// Inbound SSE reconnection state.
	lastEventID        string
	reconnectAttempts  int
	inboundCloseFunc   func()
	reconnectionConfig reconnectionConfig

	onMessage func(JSONRPCMessage)
	onClose   func()
	onError   func(error)
}

type reconnectionConfig struct {
	initialDelay time.Duration
	maxDelay     time.Duration
	growFactor   float64
	maxRetries   int
}

var _ Transport = (*HTTPTransport)(nil)

// NewHTTPTransport creates a new Streamable HTTP transport.
func NewHTTPTransport(config HTTPTransportConfig) *HTTPTransport {
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &HTTPTransport{
		config: config,
		client: client,
		reconnectionConfig: reconnectionConfig{
			initialDelay: 1 * time.Second,
			maxDelay:     30 * time.Second,
			growFactor:   1.5,
			maxRetries:   2,
		},
	}
}

// SetHandlers registers callbacks for incoming messages, transport closure,
// and transport errors.
func (t *HTTPTransport) SetHandlers(onMessage func(JSONRPCMessage), onClose func(), onError func(error)) {
	t.onMessage = onMessage
	t.onClose = onClose
	t.onError = onError
}

// Start initializes the HTTP transport and opens an optional inbound SSE stream.
func (t *HTTPTransport) Start(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.started {
		return fmt.Errorf("mcp: http transport already started")
	}

	_, cancel := context.WithCancel(ctx)
	t.cancel = cancel
	t.started = true

	go t.openInboundSSE(ctx)

	return nil
}

// Send transmits a JSON-RPC message via HTTP POST and processes the response.
func (t *HTTPTransport) Send(msg JSONRPCMessage) error {
	t.mu.Lock()
	if !t.started {
		t.mu.Unlock()
		return fmt.Errorf("mcp: http transport not started")
	}
	t.mu.Unlock()

	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mcp: marshal message: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, t.config.URL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("mcp: create request: %w", err)
	}
	t.setCommonHeaders(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")

	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("mcp: http post: %w", err)
	}
	defer func() {
		// Only close body if we don't hand it off for SSE reading.
		if resp.Body != nil {
			resp.Body.Close()
		}
	}()

	// Capture session ID from response.
	if sid := resp.Header.Get("mcp-session-id"); sid != "" {
		t.mu.Lock()
		t.sessionID = sid
		t.mu.Unlock()
	}

	// 202 Accepted — server acknowledged, no response body expected.
	if resp.StatusCode == http.StatusAccepted {
		return nil
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if err != nil {
			return fmt.Errorf("mcp: http post status %d (body unreadable: %w)", resp.StatusCode, err)
		}
		return fmt.Errorf("mcp: http post status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Notifications (no "id" field) don't expect a JSON-RPC response.
	if msg.IsNotification() {
		return nil
	}

	ct := resp.Header.Get("Content-Type")
	switch {
	case strings.Contains(ct, "application/json"):
		return t.handleJSONResponse(resp.Body)
	case strings.Contains(ct, "text/event-stream"):
		// Hand off body — prevent deferred close.
		body := resp.Body
		resp.Body = nil
		go t.readSSEStream(body)
		return nil
	default:
		return fmt.Errorf("mcp: unexpected content-type: %s", ct)
	}
}

// Close terminates the transport, sending a DELETE if a session exists.
func (t *HTTPTransport) Close() error {
	t.mu.Lock()
	sessionID := t.sessionID
	closeFn := t.inboundCloseFunc
	cancel := t.cancel
	t.mu.Unlock()

	if closeFn != nil {
		closeFn()
	}

	// Send DELETE to terminate the session.
	if sessionID != "" {
		t.deleteSession()
	}

	if cancel != nil {
		cancel()
	}

	if t.onClose != nil {
		t.onClose()
	}
	return nil
}

// setCommonHeaders applies protocol version, session ID, and custom headers.
func (t *HTTPTransport) setCommonHeaders(req *http.Request) {
	for k, v := range t.config.Headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("mcp-protocol-version", LatestProtocolVersion)

	t.mu.Lock()
	sid := t.sessionID
	t.mu.Unlock()
	if sid != "" {
		req.Header.Set("mcp-session-id", sid)
	}
}

// handleJSONResponse reads a JSON or JSON-array response body and dispatches messages.
func (t *HTTPTransport) handleJSONResponse(body io.Reader) error {
	data, err := io.ReadAll(body)
	if err != nil {
		return fmt.Errorf("mcp: read response: %w", err)
	}

	// Try as array first.
	data = bytes.TrimSpace(data)
	if len(data) > 0 && data[0] == '[' {
		var msgs []JSONRPCMessage
		if err := json.Unmarshal(data, &msgs); err != nil {
			return fmt.Errorf("mcp: parse json array response: %w", err)
		}
		for _, m := range msgs {
			if t.onMessage != nil {
				t.onMessage(m)
			}
		}
		return nil
	}

	var msg JSONRPCMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return fmt.Errorf("mcp: parse json response: %w", err)
	}
	if t.onMessage != nil {
		t.onMessage(msg)
	}
	return nil
}

// readSSEStream reads an SSE stream and dispatches JSON-RPC messages.
func (t *HTTPTransport) readSSEStream(body io.ReadCloser) {
	defer body.Close()

	scanner := bufio.NewScanner(body)
	var eventType string
	var dataLines []string

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			// Empty line = end of event.
			if len(dataLines) > 0 {
				data := strings.Join(dataLines, "\n")
				if eventType == "" || eventType == "message" {
					var msg JSONRPCMessage
					if err := json.Unmarshal([]byte(data), &msg); err != nil {
						if t.onError != nil {
							t.onError(fmt.Errorf("mcp: parse sse message: %w", err))
						}
					} else if t.onMessage != nil {
						t.onMessage(msg)
					}
				}
			}
			eventType = ""
			dataLines = nil
			continue
		}

		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		} else if strings.HasPrefix(line, "id:") {
			id := strings.TrimSpace(strings.TrimPrefix(line, "id:"))
			t.mu.Lock()
			t.lastEventID = id
			t.mu.Unlock()
		}
		// Ignore "retry:" and comment lines.
	}

	// Process any trailing event without a final blank line.
	if len(dataLines) > 0 {
		data := strings.Join(dataLines, "\n")
		if eventType == "" || eventType == "message" {
			var msg JSONRPCMessage
			if err := json.Unmarshal([]byte(data), &msg); err != nil {
				if t.onError != nil {
					t.onError(fmt.Errorf("mcp: parse sse message: %w", err))
				}
			} else if t.onMessage != nil {
				t.onMessage(msg)
			}
		}
	}
}

// openInboundSSE opens an optional server-initiated SSE stream via GET.
// Failures are non-fatal (the server may not support it).
func (t *HTTPTransport) openInboundSSE(ctx context.Context) {
	reconnect := t.doOpenInboundSSE(ctx)
	if reconnect && ctx.Err() == nil {
		t.scheduleReconnect(ctx)
	}
}

// doOpenInboundSSE performs the actual SSE GET and reads the stream.
// It returns true if a reconnection attempt should be scheduled.
func (t *HTTPTransport) doOpenInboundSSE(ctx context.Context) (reconnect bool) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, t.config.URL, nil)
	if err != nil {
		if t.onError != nil {
			t.onError(fmt.Errorf("mcp: create sse request: %w", err))
		}
		return false
	}
	t.setCommonHeaders(req)
	req.Header.Set("Accept", "text/event-stream")

	t.mu.Lock()
	lastID := t.lastEventID
	t.mu.Unlock()
	if lastID != "" {
		req.Header.Set("Last-Event-ID", lastID)
	}

	resp, err := t.client.Do(req) //nolint:bodyclose // closed by deferred resp.Body.Close below
	if err != nil {
		if ctx.Err() != nil {
			return false // context cancelled
		}
		if t.onError != nil {
			t.onError(fmt.Errorf("mcp: sse get: %w", err))
		}
		return true
	}
	defer resp.Body.Close()

	// Capture session ID.
	if sid := resp.Header.Get("mcp-session-id"); sid != "" {
		t.mu.Lock()
		t.sessionID = sid
		t.mu.Unlock()
	}

	// 405 means server doesn't support inbound SSE — that's fine.
	if resp.StatusCode == http.StatusMethodNotAllowed {
		return false
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if t.onError != nil {
			t.onError(fmt.Errorf("mcp: sse get status %d", resp.StatusCode))
		}
		return false
	}

	t.mu.Lock()
	t.inboundCloseFunc = func() { resp.Body.Close() }
	t.reconnectAttempts = 0
	t.mu.Unlock()

	t.readSSEStream(resp.Body)

	// Stream ended — attempt reconnection.
	return true
}

// scheduleReconnect retries the inbound SSE connection with exponential backoff.
func (t *HTTPTransport) scheduleReconnect(ctx context.Context) {
	t.mu.Lock()
	rc := t.reconnectionConfig
	attempt := t.reconnectAttempts
	t.reconnectAttempts++
	t.mu.Unlock()

	if rc.maxRetries > 0 && attempt >= rc.maxRetries {
		if t.onError != nil {
			t.onError(fmt.Errorf("mcp: sse reconnection attempts (%d) exceeded", rc.maxRetries))
		}
		return
	}

	delay := time.Duration(float64(rc.initialDelay) * math.Pow(rc.growFactor, float64(attempt)))
	if delay > rc.maxDelay {
		delay = rc.maxDelay
	}

	select {
	case <-time.After(delay):
		go t.openInboundSSE(ctx)
	case <-ctx.Done():
	}
}

// deleteSession sends a DELETE request to terminate the MCP session.
func (t *HTTPTransport) deleteSession() {
	req, err := http.NewRequest(http.MethodDelete, t.config.URL, nil)
	if err != nil {
		return
	}
	t.setCommonHeaders(req)

	// Best-effort, ignore errors.
	resp, err := t.client.Do(req)
	if err != nil {
		return
	}
	resp.Body.Close()
}
