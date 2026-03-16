package ai

import (
	"fmt"
	"strings"
	"sync"
)

// Registry maps model ID strings to LanguageModel instances via provider factories.
// Model IDs use "prefix:model-id" format; a default prefix handles unprefixed IDs.
type Registry struct {
	mu         sync.RWMutex
	factories  map[string]func(modelID string) LanguageModel
	defaultPfx string
}

// NewRegistry creates an empty registry.
func NewRegistry() *Registry {
	return &Registry{
		factories: make(map[string]func(modelID string) LanguageModel),
	}
}

// Register adds a provider factory for a given prefix.
// When Model("prefix:model-id") is called, the factory receives "model-id".
func (r *Registry) Register(prefix string, factory func(modelID string) LanguageModel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories[prefix] = factory
}

// SetDefault sets the default prefix used for unprefixed model IDs.
func (r *Registry) SetDefault(prefix string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.defaultPfx = prefix
}

// Model resolves a model ID string to a LanguageModel.
// Format: "prefix:model-id" or just "model-id" (uses default prefix).
// Returns an error if the prefix has no registered factory.
func (r *Registry) Model(id string) (LanguageModel, error) {
	prefix, modelID, err := r.splitID(id)
	if err != nil {
		return nil, err
	}

	r.mu.RLock()
	factory, ok := r.factories[prefix]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("registry: no provider registered for prefix %q (model %q)", prefix, id)
	}
	return factory(modelID), nil
}

// splitID splits "prefix:model-id" into (prefix, modelID).
// If no colon is present, the default prefix is used.
func (r *Registry) splitID(id string) (prefix, modelID string, err error) {
	if idx := strings.IndexByte(id, ':'); idx >= 0 {
		return id[:idx], id[idx+1:], nil
	}
	r.mu.RLock()
	dflt := r.defaultPfx
	r.mu.RUnlock()
	if dflt == "" {
		return "", "", fmt.Errorf("registry: no prefix in model ID %q and no default prefix set", id)
	}
	return dflt, id, nil
}
