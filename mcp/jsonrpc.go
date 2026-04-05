package mcp

import (
	"encoding/json"
	"fmt"
)

// JSONRPCVersion is the JSON-RPC protocol version used by MCP.
const JSONRPCVersion = "2.0"

// JSONRPCMessage is a union type representing any JSON-RPC 2.0 message:
// request, response, error, or notification.
type JSONRPCMessage struct {
	raw json.RawMessage

	// Exactly one of these will be non-nil after unmarshaling.
	Request      *JSONRPCRequest
	Response     *JSONRPCResponse
	Error        *JSONRPCError
	Notification *JSONRPCNotification
}

// JSONRPCRequest is a JSON-RPC 2.0 request with an id.
type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      JSONRPCID       `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// JSONRPCResponse is a JSON-RPC 2.0 successful response.
type JSONRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      JSONRPCID       `json:"id"`
	Result  json.RawMessage `json:"result"`
}

// JSONRPCError is a JSON-RPC 2.0 error response.
type JSONRPCError struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      JSONRPCID     `json:"id"`
	Error   JSONRPCErrObj `json:"error"`
}

// JSONRPCErrObj is the error object within a JSON-RPC error response.
type JSONRPCErrObj struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// JSONRPCNotification is a JSON-RPC 2.0 notification (no id field).
type JSONRPCNotification struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// JSONRPCID can be a string or integer, matching the JSON-RPC spec.
type JSONRPCID struct {
	str    string
	num    int64
	isStr  bool
	isNull bool
}

// StringID creates a string-typed JSON-RPC ID.
func StringID(s string) JSONRPCID {
	return JSONRPCID{str: s, isStr: true}
}

// IntID creates an integer-typed JSON-RPC ID.
func IntID(n int64) JSONRPCID {
	return JSONRPCID{num: n}
}

// MarshalJSON implements json.Marshaler for JSONRPCID.
func (id JSONRPCID) MarshalJSON() ([]byte, error) {
	if id.isNull {
		return []byte("null"), nil
	}
	if id.isStr {
		return json.Marshal(id.str)
	}
	return json.Marshal(id.num)
}

// UnmarshalJSON implements json.Unmarshaler for JSONRPCID.
func (id *JSONRPCID) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		id.isNull = true
		return nil
	}
	// Try string first.
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		id.str = s
		id.isStr = true
		return nil
	}
	// Try integer.
	var n int64
	if err := json.Unmarshal(data, &n); err == nil {
		id.num = n
		return nil
	}
	return fmt.Errorf("jsonrpc: id must be string or integer, got %s", string(data))
}

// String returns a human-readable representation of the ID.
func (id JSONRPCID) String() string {
	if id.isNull {
		return "null"
	}
	if id.isStr {
		return id.str
	}
	return fmt.Sprintf("%d", id.num)
}

// MarshalJSON serializes a JSONRPCMessage to its wire format.
func (m JSONRPCMessage) MarshalJSON() ([]byte, error) {
	switch {
	case m.Request != nil:
		return json.Marshal(m.Request)
	case m.Response != nil:
		return json.Marshal(m.Response)
	case m.Error != nil:
		return json.Marshal(m.Error)
	case m.Notification != nil:
		return json.Marshal(m.Notification)
	case m.raw != nil:
		return m.raw, nil
	default:
		return nil, fmt.Errorf("jsonrpc: empty message")
	}
}

// UnmarshalJSON parses a JSON-RPC 2.0 message and classifies it.
func (m *JSONRPCMessage) UnmarshalJSON(data []byte) error {
	m.raw = append(json.RawMessage(nil), data...)

	// Probe the shape to determine the message type.
	var probe struct {
		JSONRPC string           `json:"jsonrpc"`
		ID      *json.RawMessage `json:"id"`
		Method  *string          `json:"method"`
		Result  *json.RawMessage `json:"result"`
		Error   *json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return fmt.Errorf("jsonrpc: unmarshal probe: %w", err)
	}

	if probe.JSONRPC != JSONRPCVersion {
		return fmt.Errorf("jsonrpc: unexpected version %q", probe.JSONRPC)
	}

	switch {
	case probe.ID != nil && probe.Error != nil:
		m.Error = &JSONRPCError{}
		return json.Unmarshal(data, m.Error)
	case probe.ID != nil && probe.Result != nil:
		m.Response = &JSONRPCResponse{}
		return json.Unmarshal(data, m.Response)
	case probe.ID != nil && probe.Method != nil:
		m.Request = &JSONRPCRequest{}
		return json.Unmarshal(data, m.Request)
	case probe.Method != nil:
		m.Notification = &JSONRPCNotification{}
		return json.Unmarshal(data, m.Notification)
	default:
		return fmt.Errorf("jsonrpc: cannot classify message: %s", string(data))
	}
}

// IsRequest reports whether the message is a JSON-RPC request.
func (m JSONRPCMessage) IsRequest() bool { return m.Request != nil }

// IsResponse reports whether the message is a JSON-RPC response.
func (m JSONRPCMessage) IsResponse() bool { return m.Response != nil }

// IsError reports whether the message is a JSON-RPC error.
func (m JSONRPCMessage) IsError() bool { return m.Error != nil }

// IsNotification reports whether the message is a JSON-RPC notification.
func (m JSONRPCMessage) IsNotification() bool { return m.Notification != nil }

// NewRequest creates a JSON-RPC request message.
func NewRequest(id JSONRPCID, method string, params json.RawMessage) JSONRPCMessage {
	return JSONRPCMessage{
		Request: &JSONRPCRequest{
			JSONRPC: JSONRPCVersion,
			ID:      id,
			Method:  method,
			Params:  params,
		},
	}
}

// NewResponse creates a JSON-RPC response message.
func NewResponse(id JSONRPCID, result json.RawMessage) JSONRPCMessage {
	return JSONRPCMessage{
		Response: &JSONRPCResponse{
			JSONRPC: JSONRPCVersion,
			ID:      id,
			Result:  result,
		},
	}
}

// NewNotification creates a JSON-RPC notification message.
func NewNotification(method string, params json.RawMessage) JSONRPCMessage {
	return JSONRPCMessage{
		Notification: &JSONRPCNotification{
			JSONRPC: JSONRPCVersion,
			Method:  method,
			Params:  params,
		},
	}
}
