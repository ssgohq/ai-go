package ai

import "sync"

// ModelPrice holds per-million-token pricing for a model.
type ModelPrice struct {
	PromptPer1M     float64
	CompletionPer1M float64
	CachedPer1M     float64 // prompt cache read pricing (if applicable)
}

// StepCost holds the cost breakdown for a single step.
type StepCost struct {
	Model            string
	PromptTokens     int
	CompletionTokens int
	CostUSD          float64
}

// CostTracker accumulates cost across steps.
type CostTracker struct {
	mu       sync.Mutex
	costs    []StepCost
	totalUSD float64
}

// NewCostTracker creates a new CostTracker.
func NewCostTracker() *CostTracker {
	return &CostTracker{}
}

// Add records a step's cost.
func (ct *CostTracker) Add(cost StepCost) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.costs = append(ct.costs, cost)
	ct.totalUSD += cost.CostUSD
}

// TotalCost returns the accumulated cost in USD.
func (ct *CostTracker) TotalCost() float64 {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.totalUSD
}

// Steps returns a copy of all step costs.
func (ct *CostTracker) Steps() []StepCost {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	out := make([]StepCost, len(ct.costs))
	copy(out, ct.costs)
	return out
}

// CalculateCost computes USD cost from usage and pricing.
func CalculateCost(model string, usage Usage, price ModelPrice) StepCost {
	promptCost := float64(usage.PromptTokens) * price.PromptPer1M / 1_000_000
	completionCost := float64(usage.CompletionTokens) * price.CompletionPer1M / 1_000_000
	cachedCost := float64(usage.CacheReadTokens) * price.CachedPer1M / 1_000_000
	return StepCost{
		Model:            model,
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		CostUSD:          promptCost + completionCost + cachedCost,
	}
}

// DefaultModelPricing contains known model prices (per 1M tokens).
// Update at runtime via SetModelPrice.
var defaultPricing = struct {
	mu     sync.RWMutex
	prices map[string]ModelPrice
}{
	prices: map[string]ModelPrice{
		"gpt-4o":                  {PromptPer1M: 2.50, CompletionPer1M: 10.00},
		"gpt-4o-mini":             {PromptPer1M: 0.15, CompletionPer1M: 0.60},
		"gpt-4.1":                 {PromptPer1M: 2.00, CompletionPer1M: 8.00},
		"gpt-4.1-mini":            {PromptPer1M: 0.40, CompletionPer1M: 1.60},
		"gpt-4.1-nano":            {PromptPer1M: 0.10, CompletionPer1M: 0.40},
		"o3":                      {PromptPer1M: 2.00, CompletionPer1M: 8.00},
		"o3-mini":                 {PromptPer1M: 1.10, CompletionPer1M: 4.40},
		"o4-mini":                 {PromptPer1M: 1.10, CompletionPer1M: 4.40},
		"claude-4-sonnet":         {PromptPer1M: 3.00, CompletionPer1M: 15.00, CachedPer1M: 0.30},
		"claude-4-opus":           {PromptPer1M: 15.00, CompletionPer1M: 75.00, CachedPer1M: 1.50},
		"claude-3.7-sonnet":       {PromptPer1M: 3.00, CompletionPer1M: 15.00, CachedPer1M: 0.30},
		"claude-3.5-sonnet":       {PromptPer1M: 3.00, CompletionPer1M: 15.00, CachedPer1M: 0.30},
		"gemini-2.5-pro":          {PromptPer1M: 1.25, CompletionPer1M: 10.00},
		"gemini-2.5-flash":        {PromptPer1M: 0.15, CompletionPer1M: 0.60},
		"gemini-2.0-flash":        {PromptPer1M: 0.10, CompletionPer1M: 0.40},
	},
}

// GetModelPrice returns the pricing for a model ID. Returns zero price if unknown.
func GetModelPrice(modelID string) (ModelPrice, bool) {
	defaultPricing.mu.RLock()
	defer defaultPricing.mu.RUnlock()
	p, ok := defaultPricing.prices[modelID]
	return p, ok
}

// SetModelPrice updates or adds pricing for a model at runtime.
func SetModelPrice(modelID string, price ModelPrice) {
	defaultPricing.mu.Lock()
	defer defaultPricing.mu.Unlock()
	defaultPricing.prices[modelID] = price
}
