package uistream

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// TestFixture_TextOnly verifies the text-only fixture is well-formed and contains required chunks.
func TestFixture_TextOnly(t *testing.T) {
	lines := readFixtureLines(t, "testdata/text-only.jsonl")
	assertAnyLineContains(t, lines, `"type":"start"`)
	assertAnyLineContains(t, lines, `"type":"start-step"`)
	assertAnyLineContains(t, lines, `"type":"text-start"`)
	assertAnyLineContains(t, lines, `"type":"text-delta"`)
	assertAnyLineContains(t, lines, `"type":"text-end"`)
	assertAnyLineContains(t, lines, `"type":"finish-step"`)
	assertAnyLineContains(t, lines, `"type":"finish"`)
	assertAnyLineContains(t, lines, "[DONE]")
}

// TestFixture_ReasoningWithSources verifies reasoning + source chunks are present.
func TestFixture_ReasoningWithSources(t *testing.T) {
	lines := readFixtureLines(t, "testdata/reasoning-with-sources.jsonl")
	assertAnyLineContains(t, lines, `"type":"start"`)
	assertAnyLineContains(t, lines, `"type":"reasoning-start"`)
	assertAnyLineContains(t, lines, `"type":"reasoning-delta"`)
	assertAnyLineContains(t, lines, `"type":"reasoning-end"`)
	assertAnyLineContains(t, lines, `"type":"text-start"`)
	assertAnyLineContains(t, lines, `"type":"text-delta"`)
	assertAnyLineContains(t, lines, `"type":"source"`)
	assertAnyLineContains(t, lines, `"type":"sources"`)
	assertAnyLineContains(t, lines, `"type":"finish"`)
	assertAnyLineContains(t, lines, "[DONE]")
}

// TestFixture_ToolCallLifecycle verifies the full tool call sequence including custom data chunks.
func TestFixture_ToolCallLifecycle(t *testing.T) {
	lines := readFixtureLines(t, "testdata/tool-call-lifecycle.jsonl")
	assertAnyLineContains(t, lines, `"type":"start"`)
	assertAnyLineContains(t, lines, `"type":"tool-input-start"`)
	assertAnyLineContains(t, lines, `"type":"tool-input-delta"`)
	assertAnyLineContains(t, lines, `"type":"tool-input-available"`)
	assertAnyLineContains(t, lines, `"type":"tool-output-available"`)
	assertAnyLineContains(t, lines, `"type":"data-document-references"`)
	assertAnyLineContains(t, lines, `"type":"data-usage"`)
	assertAnyLineContains(t, lines, `"type":"finish"`)
	assertAnyLineContains(t, lines, "[DONE]")
}

// TestFixture_DeepThinkingFull verifies the deep-thinking scenario covers all planned chunk types.
func TestFixture_DeepThinkingFull(t *testing.T) {
	lines := readFixtureLines(t, "testdata/deep-thinking-full.jsonl")
	assertAnyLineContains(t, lines, `"type":"start"`)
	assertAnyLineContains(t, lines, `"type":"data-plan"`)
	assertAnyLineContains(t, lines, `"type":"data-steps"`)
	assertAnyLineContains(t, lines, `"type":"reasoning-start"`)
	assertAnyLineContains(t, lines, `"type":"reasoning-delta"`)
	assertAnyLineContains(t, lines, `"type":"reasoning-end"`)
	assertAnyLineContains(t, lines, `"type":"tool-input-start"`)
	assertAnyLineContains(t, lines, `"type":"tool-output-available"`)
	assertAnyLineContains(t, lines, `"type":"text-start"`)
	assertAnyLineContains(t, lines, `"type":"text-delta"`)
	assertAnyLineContains(t, lines, `"type":"source"`)
	assertAnyLineContains(t, lines, `"type":"sources"`)
	assertAnyLineContains(t, lines, `"type":"data-suggested-questions"`)
	assertAnyLineContains(t, lines, `"type":"data-usage"`)
	assertAnyLineContains(t, lines, `"type":"finish"`)
	assertAnyLineContains(t, lines, "[DONE]")
}

// TestFixture_ErrorMidStream verifies the error fixture terminates with an error chunk and no finish.
func TestFixture_ErrorMidStream(t *testing.T) {
	lines := readFixtureLines(t, "testdata/error-mid-stream.jsonl")
	assertAnyLineContains(t, lines, `"type":"start"`)
	assertAnyLineContains(t, lines, `"type":"text-delta"`)
	assertAnyLineContains(t, lines, `"type":"error"`)
	assertNoneContains(t, lines, `"type":"finish"`)
	assertNoneContains(t, lines, "[DONE]")
}

// --- helpers ---

func readFixtureLines(t *testing.T, path string) []string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open fixture %s: %v", path, err)
	}
	defer f.Close()

	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line != "" {
			lines = append(lines, line)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan fixture %s: %v", path, err)
	}
	return lines
}

func assertAnyLineContains(t *testing.T, lines []string, want string) {
	t.Helper()
	for _, l := range lines {
		if strings.Contains(l, want) {
			return
		}
	}
	t.Errorf("no fixture line contains %q", want)
}

func assertNoneContains(t *testing.T, lines []string, want string) {
	t.Helper()
	for _, l := range lines {
		if strings.Contains(l, want) {
			t.Errorf("fixture line contains %q but should not: %s", want, l)
			return
		}
	}
}
