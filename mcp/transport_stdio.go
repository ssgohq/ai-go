package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
)

// StdioConfig configures a stdio-based MCP transport that communicates with
// a subprocess via newline-delimited JSON on stdin/stdout.
type StdioConfig struct {
	// Command is the executable to run.
	Command string
	// Args are command-line arguments.
	Args []string
	// Env is additional environment variables ("KEY=VALUE" format).
	// The subprocess inherits the parent environment plus these entries.
	Env []string
	// Dir is the working directory for the subprocess.
	Dir string
}

// StdioTransport implements Transport by spawning a subprocess and exchanging
// newline-delimited JSON-RPC messages over its stdin and stdout.
type StdioTransport struct {
	config StdioConfig

	mu      sync.Mutex
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	cancel  context.CancelFunc
	started bool

	onMessage func(JSONRPCMessage)
	onClose   func()
	onError   func(error)
}

var _ Transport = (*StdioTransport)(nil)

// NewStdioTransport creates a new stdio transport with the given configuration.
func NewStdioTransport(config StdioConfig) *StdioTransport {
	return &StdioTransport{config: config}
}

// SetHandlers registers callbacks for incoming messages, transport closure,
// and transport errors.
func (t *StdioTransport) SetHandlers(onMessage func(JSONRPCMessage), onClose func(), onError func(error)) {
	t.onMessage = onMessage
	t.onClose = onClose
	t.onError = onError
}

// Start spawns the subprocess and begins reading from its stdout.
func (t *StdioTransport) Start(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.started {
		return fmt.Errorf("mcp: stdio transport already started")
	}

	procCtx, cancel := context.WithCancel(ctx)
	t.cancel = cancel

	cmd := exec.CommandContext(procCtx, t.config.Command, t.config.Args...)
	if t.config.Dir != "" {
		cmd.Dir = t.config.Dir
	}
	if len(t.config.Env) > 0 {
		cmd.Env = append(cmd.Environ(), t.config.Env...)
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return fmt.Errorf("mcp: stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return fmt.Errorf("mcp: stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		cancel()
		return fmt.Errorf("mcp: start process: %w", err)
	}

	t.cmd = cmd
	t.stdin = stdin
	t.started = true

	// Read stdout in a goroutine.
	go t.readLoop(procCtx, stdout, cmd)

	return nil
}

// readLoop reads newline-delimited JSON from the subprocess stdout.
func (t *StdioTransport) readLoop(ctx context.Context, r io.Reader, cmd *exec.Cmd) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024) // up to 10MB lines
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var msg JSONRPCMessage
		if err := json.Unmarshal(line, &msg); err != nil {
			if t.onError != nil {
				t.onError(fmt.Errorf("mcp: parse message: %w", err))
			}
			continue
		}
		if t.onMessage != nil {
			t.onMessage(msg)
		}
	}
	if err := scanner.Err(); err != nil {
		if t.onError != nil && ctx.Err() == nil {
			t.onError(fmt.Errorf("mcp: read stdout: %w", err))
		}
	}
	// Wait for the process to exit.
	_ = cmd.Wait()
	if t.onClose != nil {
		t.onClose()
	}
}

// Send writes a JSON-RPC message as newline-delimited JSON to the subprocess stdin.
func (t *StdioTransport) Send(msg JSONRPCMessage) error {
	t.mu.Lock()
	w := t.stdin
	t.mu.Unlock()

	if w == nil {
		return fmt.Errorf("mcp: stdio transport not started")
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mcp: marshal message: %w", err)
	}
	data = append(data, '\n')
	_, err = w.Write(data)
	if err != nil {
		return fmt.Errorf("mcp: write stdin: %w", err)
	}
	return nil
}

// Close terminates the subprocess and cleans up resources.
func (t *StdioTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.cancel != nil {
		t.cancel()
		t.cancel = nil
	}
	if t.stdin != nil {
		t.stdin.Close()
		t.stdin = nil
	}
	return nil
}
