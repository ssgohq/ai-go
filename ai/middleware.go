package ai

// LanguageModelMiddleware wraps a LanguageModel to add cross-cutting behavior.
// Each middleware receives the inner model and returns a new model that
// delegates to it after (optionally) modifying the request or response stream.
type LanguageModelMiddleware func(model LanguageModel) LanguageModel

// WrapLanguageModel applies middlewares to model in order (first middleware is outermost).
// Example:
//
//	wrapped := ai.WrapLanguageModel(model, loggingMiddleware, cachingMiddleware)
func WrapLanguageModel(model LanguageModel, middlewares ...LanguageModelMiddleware) LanguageModel {
	for i := len(middlewares) - 1; i >= 0; i-- {
		model = middlewares[i](model)
	}
	return model
}

// WithMiddleware returns an Option that wraps the request's model with the given
// middlewares. The middlewares are stored and applied after model resolution, so
// this works correctly with both WithModel and Runtime.WithDefaultModel.
func WithMiddleware(middlewares ...LanguageModelMiddleware) Option {
	return func(r *GenerateTextRequest) {
		r.Middlewares = append(r.Middlewares, middlewares...)
	}
}
