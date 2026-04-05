package mcp

import (
	"encoding/json"
	"testing"
)

func TestJSONRPCRequest_MarshalUnmarshal(t *testing.T) {
	req := &JSONRPCRequest{
		JSONRPC: JSONRPCVersion,
		ID:      IntID(1),
		Method:  "tools/list",
		Params:  json.RawMessage(`{"cursor":"abc"}`),
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got JSONRPCRequest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.JSONRPC != JSONRPCVersion {
		t.Errorf("jsonrpc = %q, want %q", got.JSONRPC, JSONRPCVersion)
	}
	if got.ID.String() != "1" {
		t.Errorf("id = %q, want %q", got.ID.String(), "1")
	}
	if got.Method != "tools/list" {
		t.Errorf("method = %q, want %q", got.Method, "tools/list")
	}
	if string(got.Params) != `{"cursor":"abc"}` {
		t.Errorf("params = %s, want %s", got.Params, `{"cursor":"abc"}`)
	}
}

func TestJSONRPCResponse_MarshalUnmarshal(t *testing.T) {
	resp := &JSONRPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      StringID("req-42"),
		Result:  json.RawMessage(`{"tools":[]}`),
	}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got JSONRPCResponse
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.ID.String() != "req-42" {
		t.Errorf("id = %q, want %q", got.ID.String(), "req-42")
	}
	if string(got.Result) != `{"tools":[]}` {
		t.Errorf("result = %s, want %s", got.Result, `{"tools":[]}`)
	}
}

func TestJSONRPCError_MarshalUnmarshal(t *testing.T) {
	errResp := &JSONRPCError{
		JSONRPC: JSONRPCVersion,
		ID:      IntID(5),
		Error: JSONRPCErrObj{
			Code:    -32601,
			Message: "Method not found",
		},
	}
	data, err := json.Marshal(errResp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got JSONRPCError
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Error.Code != -32601 {
		t.Errorf("error.code = %d, want %d", got.Error.Code, -32601)
	}
	if got.Error.Message != "Method not found" {
		t.Errorf("error.message = %q, want %q", got.Error.Message, "Method not found")
	}
}

func TestJSONRPCNotification_MarshalUnmarshal(t *testing.T) {
	notif := &JSONRPCNotification{
		JSONRPC: JSONRPCVersion,
		Method:  "notifications/initialized",
	}
	data, err := json.Marshal(notif)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got JSONRPCNotification
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Method != "notifications/initialized" {
		t.Errorf("method = %q, want %q", got.Method, "notifications/initialized")
	}
	if got.Params != nil {
		t.Errorf("params = %s, want nil", got.Params)
	}
}

func TestJSONRPCMessage_RoundTrip_Request(t *testing.T) {
	msg := NewRequest(IntID(1), "tools/list", json.RawMessage(`{}`))
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got JSONRPCMessage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !got.IsRequest() {
		t.Fatal("expected IsRequest() = true")
	}
	if got.Request.Method != "tools/list" {
		t.Errorf("method = %q, want %q", got.Request.Method, "tools/list")
	}
}

func TestJSONRPCMessage_RoundTrip_Response(t *testing.T) {
	msg := NewResponse(IntID(1), json.RawMessage(`{"tools":[]}`))
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got JSONRPCMessage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !got.IsResponse() {
		t.Fatal("expected IsResponse() = true")
	}
	if string(got.Response.Result) != `{"tools":[]}` {
		t.Errorf("result = %s, want %s", got.Response.Result, `{"tools":[]}`)
	}
}

func TestJSONRPCMessage_RoundTrip_Error(t *testing.T) {
	msg := JSONRPCMessage{
		Error: &JSONRPCError{
			JSONRPC: JSONRPCVersion,
			ID:      IntID(1),
			Error: JSONRPCErrObj{
				Code:    -32600,
				Message: "Invalid Request",
			},
		},
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got JSONRPCMessage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !got.IsError() {
		t.Fatal("expected IsError() = true")
	}
	if got.Error.Error.Code != -32600 {
		t.Errorf("error.code = %d, want %d", got.Error.Error.Code, -32600)
	}
}

func TestJSONRPCMessage_RoundTrip_Notification(t *testing.T) {
	msg := NewNotification("notifications/initialized", nil)
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got JSONRPCMessage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !got.IsNotification() {
		t.Fatal("expected IsNotification() = true")
	}
	if got.Notification.Method != "notifications/initialized" {
		t.Errorf("method = %q, want %q", got.Notification.Method, "notifications/initialized")
	}
}

func TestJSONRPCID_StringType(t *testing.T) {
	id := StringID("abc-123")
	data, err := json.Marshal(id)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(data) != `"abc-123"` {
		t.Errorf("marshal = %s, want %s", data, `"abc-123"`)
	}

	var got JSONRPCID
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.String() != "abc-123" {
		t.Errorf("String() = %q, want %q", got.String(), "abc-123")
	}
}

func TestJSONRPCID_IntType(t *testing.T) {
	id := IntID(42)
	data, err := json.Marshal(id)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(data) != "42" {
		t.Errorf("marshal = %s, want %s", data, "42")
	}

	var got JSONRPCID
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.String() != "42" {
		t.Errorf("String() = %q, want %q", got.String(), "42")
	}
}

func TestJSONRPCID_NullType(t *testing.T) {
	var got JSONRPCID
	if err := json.Unmarshal([]byte("null"), &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.String() != "null" {
		t.Errorf("String() = %q, want %q", got.String(), "null")
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(data) != "null" {
		t.Errorf("marshal = %s, want null", data)
	}
}

func TestJSONRPCMessage_EmptyMessage_MarshalError(t *testing.T) {
	msg := JSONRPCMessage{}
	_, err := json.Marshal(msg)
	if err == nil {
		t.Fatal("expected error marshaling empty message")
	}
}

func TestJSONRPCMessage_BadVersion_UnmarshalError(t *testing.T) {
	data := []byte(`{"jsonrpc":"1.0","method":"test"}`)
	var msg JSONRPCMessage
	if err := json.Unmarshal(data, &msg); err == nil {
		t.Fatal("expected error for wrong jsonrpc version")
	}
}
