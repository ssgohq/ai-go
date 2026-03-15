package ai

import "testing"

func TestStepCountIs(t *testing.T) {
	cond := StepCountIs(3)

	if cond(2, &StepResult{}) {
		t.Error("should not stop at step 2 when limit is 3")
	}
	if !cond(3, &StepResult{}) {
		t.Error("should stop at step 3 when limit is 3")
	}
	if !cond(4, &StepResult{}) {
		t.Error("should stop at step 4 when limit is 3")
	}
}

func TestNever(t *testing.T) {
	cond := Never()
	for step := 1; step <= 100; step++ {
		if cond(step, &StepResult{}) {
			t.Errorf("Never() should not stop at step %d", step)
		}
	}
}

func TestOutputObject(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	o := OutputObject(schema)
	if o.Type != "object" {
		t.Errorf("expected type object, got %s", o.Type)
	}
	if o.Schema == nil {
		t.Error("expected schema to be set")
	}
}

func TestOutputArray(t *testing.T) {
	item := map[string]any{"type": "string"}
	o := OutputArray(item)
	if o.Type != "array" {
		t.Errorf("expected type array, got %s", o.Type)
	}
	items, ok := o.Schema["items"]
	if !ok {
		t.Error("expected items key in schema")
	}
	itemMap, ok := items.(map[string]any)
	if !ok || itemMap["type"] != "string" {
		t.Error("expected items to be string schema")
	}
}

func TestOutputText(t *testing.T) {
	o := OutputText()
	if o.Type != "text" {
		t.Errorf("expected type text, got %s", o.Type)
	}
	if o.Schema != nil {
		t.Error("expected schema to be nil for text output")
	}
}
