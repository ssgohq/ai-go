package openaichat

// CapabilityFlags declares optional features a provider supports.
type CapabilityFlags struct {
	// SupportsStructuredOutput indicates the provider accepts json_schema response_format.
	SupportsStructuredOutput bool
	// SupportsStreamUsage indicates the provider emits usage in streaming chunks.
	SupportsStreamUsage bool
}
