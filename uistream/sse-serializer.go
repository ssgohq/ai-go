package uistream

import (
	"encoding/json"
	"fmt"
	"io"
)

// WriteSSE serializes a single Chunk to SSE format on w.
// The output is a "data: <json>\n\n" line.
// If the chunk type is ChunkFinish, a trailing "data: [DONE]\n\n" is also emitted.
func WriteSSE(w io.Writer, c Chunk) {
	payload := make(map[string]any, len(c.Fields)+1)
	for k, v := range c.Fields {
		payload[k] = v
	}
	payload["type"] = c.Type

	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", b)

	if c.Type == ChunkFinish {
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}
}

// WriteSSEStream reads all chunks from the channel and writes each as SSE to w.
// It blocks until the channel is closed.
func WriteSSEStream(w io.Writer, chunks <-chan Chunk) {
	for c := range chunks {
		WriteSSE(w, c)
	}
}
