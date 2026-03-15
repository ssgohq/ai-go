package schema

import "testing"

func TestObject(t *testing.T) {
	s := Object(map[string]any{
		"name": String("the name"),
	}, "name")

	if s["type"] != "object" {
		t.Errorf("expected object type, got %v", s["type"])
	}
	req, ok := s["required"].([]string)
	if !ok || len(req) != 1 || req[0] != "name" {
		t.Errorf("unexpected required: %v", s["required"])
	}
}

func TestArray(t *testing.T) {
	s := Array(String("item"))
	if s["type"] != "array" {
		t.Errorf("expected array type, got %v", s["type"])
	}
	items, ok := s["items"].(map[string]any)
	if !ok || items["type"] != "string" {
		t.Errorf("unexpected items: %v", s["items"])
	}
}

func TestString(t *testing.T) {
	s := String("a description")
	if s["type"] != "string" {
		t.Errorf("expected string type")
	}
	if s["description"] != "a description" {
		t.Errorf("expected description")
	}
}

func TestStringNoDescription(t *testing.T) {
	s := String("")
	if _, ok := s["description"]; ok {
		t.Error("description should be omitted when empty")
	}
}

func TestNumber(t *testing.T) {
	s := Number("a number")
	if s["type"] != "number" {
		t.Errorf("expected number type")
	}
}

func TestInteger(t *testing.T) {
	s := Integer("")
	if s["type"] != "integer" {
		t.Errorf("expected integer type")
	}
}

func TestBoolean(t *testing.T) {
	s := Boolean("flag")
	if s["type"] != "boolean" {
		t.Errorf("expected boolean type")
	}
}

func TestEnum(t *testing.T) {
	s := Enum("a", "b", "c")
	if s["type"] != "string" {
		t.Errorf("expected string type for enum")
	}
	vals, ok := s["enum"].([]any)
	if !ok || len(vals) != 3 {
		t.Errorf("expected 3 enum values, got %v", s["enum"])
	}
}
