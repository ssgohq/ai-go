package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// DefineTool creates a Tool from a typed handler function.
// T must be a struct; its fields become the tool's input schema via reflection.
//
// Supported struct field tags:
//   - json:"name"        — JSON key for the field (use json:"-" to exclude)
//   - description:"..."  — schema description for the field
//   - enum:"a,b,c"       — restricts field to enumerated string values
//
// Pointer fields are optional; non-pointer fields are required.
func DefineTool[T any](
	name, description string,
	fn func(ctx context.Context, input T) (string, error),
) Tool {
	var zero T
	schema := schemaFromStruct(reflect.TypeOf(zero))
	return Tool{
		Name:        name,
		Description: description,
		InputSchema: schema,
		Execute: func(ctx context.Context, argsJSON string) (string, error) {
			var input T
			if err := json.Unmarshal([]byte(argsJSON), &input); err != nil {
				return "", fmt.Errorf("DefineTool %q: unmarshal args: %w", name, err)
			}
			return fn(ctx, input)
		},
	}
}

// schemaFromStruct builds a JSON Schema object map from a struct type.
func schemaFromStruct(t reflect.Type) map[string]any {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	properties := make(map[string]any)
	var required []string

	for i := range t.NumField() {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		jsonKey := jsonFieldName(field)
		if jsonKey == "" {
			continue // json:"-"
		}

		prop := fieldSchema(field)
		properties[jsonKey] = prop

		// Non-pointer fields are required.
		if field.Type.Kind() != reflect.Ptr {
			required = append(required, jsonKey)
		}
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

// jsonFieldName returns the JSON key for a struct field, or "" to skip it.
func jsonFieldName(field reflect.StructField) string {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return ""
	}
	if tag == "" {
		return field.Name
	}
	name, _, _ := strings.Cut(tag, ",")
	if name == "-" {
		return ""
	}
	if name == "" {
		return field.Name
	}
	return name
}

// fieldSchema builds a JSON Schema property map for a single struct field.
func fieldSchema(field reflect.StructField) map[string]any {
	ft := field.Type
	for ft.Kind() == reflect.Ptr {
		ft = ft.Elem()
	}

	prop := goTypeToSchema(ft)

	if desc := field.Tag.Get("description"); desc != "" {
		prop["description"] = desc
	}

	if enumTag := field.Tag.Get("enum"); enumTag != "" {
		parts := strings.Split(enumTag, ",")
		vals := make([]any, len(parts))
		for i, v := range parts {
			vals[i] = strings.TrimSpace(v)
		}
		prop["enum"] = vals
		prop["type"] = "string"
	}

	return prop
}

// goTypeToSchema converts a reflect.Type to a JSON Schema type map.
func goTypeToSchema(t reflect.Type) map[string]any {
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice:
		itemSchema := goTypeToSchema(t.Elem())
		return map[string]any{"type": "array", "items": itemSchema}
	case reflect.Struct:
		return schemaFromStruct(t)
	default:
		return map[string]any{"type": "string"}
	}
}
