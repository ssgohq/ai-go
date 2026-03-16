package ai_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/open-ai-sdk/ai-go/ai"
)

// weatherInput is a sample typed input struct for DefineTool tests.
type weatherInput struct {
	Location string  `json:"location" description:"City and state"`
	Unit     string  `json:"unit,omitempty" description:"Temperature unit" enum:"celsius,fahrenheit"`
	MaxTemp  float64 `json:"maxTemp,omitempty" description:"Maximum temperature threshold"`
	Verbose  bool    `json:"verbose" description:"Include extra details"`
}

// optionalInput has a pointer field that should be optional.
type optionalInput struct {
	Required string  `json:"required"`
	Optional *string `json:"optional,omitempty"`
}

// skippedInput has a field tagged json:"-".
type skippedInput struct {
	Visible string `json:"visible"`
	Hidden  string `json:"-"`
}

func TestDefineTool_Schema_RequiredFields(t *testing.T) {
	tool := ai.DefineTool[weatherInput](
		"get_weather",
		"Get the current weather",
		func(_ context.Context, _ weatherInput) (string, error) { return "", nil },
	)

	if tool.Name != "get_weather" {
		t.Errorf("Name: got %q, want %q", tool.Name, "get_weather")
	}
	if tool.Description != "Get the current weather" {
		t.Errorf("Description: got %q", tool.Description)
	}

	schemaType, _ := tool.InputSchema["type"].(string)
	if schemaType != "object" {
		t.Errorf("schema type: got %q, want %q", schemaType, "object")
	}

	props, ok := tool.InputSchema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema missing properties")
	}

	// location and verbose are non-pointer → required
	required, _ := tool.InputSchema["required"].([]string)
	requiredSet := make(map[string]bool, len(required))
	for _, r := range required {
		requiredSet[r] = true
	}

	if !requiredSet["location"] {
		t.Error("expected 'location' to be required")
	}
	if !requiredSet["verbose"] {
		t.Error("expected 'verbose' to be required")
	}

	// unit and maxTemp are non-pointer but omitempty → still non-pointer, so required
	// (omitempty is a JSON serialisation hint, not a pointer)
	// The schema should have at least location property
	if _, exists := props["location"]; !exists {
		t.Error("expected 'location' in properties")
	}
}

func TestDefineTool_Schema_PointerFieldsOptional(t *testing.T) {
	tool := ai.DefineTool[optionalInput](
		"test_optional",
		"Test optional fields",
		func(_ context.Context, _ optionalInput) (string, error) { return "", nil },
	)

	required, _ := tool.InputSchema["required"].([]string)
	requiredSet := make(map[string]bool, len(required))
	for _, r := range required {
		requiredSet[r] = true
	}

	if !requiredSet["required"] {
		t.Error("expected 'required' field to be in required list")
	}
	if requiredSet["optional"] {
		t.Error("expected pointer 'optional' field to NOT be in required list")
	}
}

func TestDefineTool_Schema_SkippedField(t *testing.T) {
	tool := ai.DefineTool[skippedInput](
		"test_skip",
		"Test skip",
		func(_ context.Context, _ skippedInput) (string, error) { return "", nil },
	)

	props, ok := tool.InputSchema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema missing properties")
	}

	if _, exists := props["Hidden"]; exists {
		t.Error("json:\"-\" field should be excluded from schema")
	}
	if _, exists := props["visible"]; !exists {
		t.Error("expected 'visible' in properties")
	}
}

func TestDefineTool_Schema_EnumTag(t *testing.T) {
	tool := ai.DefineTool[weatherInput](
		"get_weather",
		"Get weather",
		func(_ context.Context, _ weatherInput) (string, error) { return "", nil },
	)

	props, _ := tool.InputSchema["properties"].(map[string]any)
	unitProp, ok := props["unit"].(map[string]any)
	if !ok {
		t.Fatal("expected 'unit' property")
	}

	enumVals, ok := unitProp["enum"].([]any)
	if !ok {
		t.Fatal("expected enum in 'unit' property")
	}
	if len(enumVals) != 2 {
		t.Errorf("expected 2 enum values, got %d", len(enumVals))
	}
}

func TestDefineTool_Schema_DescriptionTag(t *testing.T) {
	tool := ai.DefineTool[weatherInput](
		"get_weather",
		"Get weather",
		func(_ context.Context, _ weatherInput) (string, error) { return "", nil },
	)

	props, _ := tool.InputSchema["properties"].(map[string]any)
	locProp, ok := props["location"].(map[string]any)
	if !ok {
		t.Fatal("expected 'location' property")
	}

	desc, _ := locProp["description"].(string)
	if desc == "" {
		t.Error("expected description tag to be reflected in schema")
	}
}

func TestDefineTool_Execute_UnmarshalsAndCalls(t *testing.T) {
	called := false
	var capturedInput weatherInput

	tool := ai.DefineTool[weatherInput](
		"get_weather",
		"Get weather",
		func(_ context.Context, input weatherInput) (string, error) {
			called = true
			capturedInput = input
			return "sunny, 72°F", nil
		},
	)

	args := weatherInput{Location: "San Francisco, CA", Unit: "fahrenheit"}
	argsJSON, _ := json.Marshal(args)

	result, err := tool.Execute(context.Background(), string(argsJSON))
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if !called {
		t.Error("handler was not called")
	}
	if capturedInput.Location != "San Francisco, CA" {
		t.Errorf("Location: got %q", capturedInput.Location)
	}
	if result != "sunny, 72°F" {
		t.Errorf("result: got %q", result)
	}
}

func TestDefineTool_Execute_InvalidJSON(t *testing.T) {
	tool := ai.DefineTool[weatherInput](
		"get_weather",
		"Get weather",
		func(_ context.Context, _ weatherInput) (string, error) { return "", nil },
	)

	_, err := tool.Execute(context.Background(), "not-valid-json")
	if err == nil {
		t.Error("expected error on invalid JSON, got nil")
	}
}
