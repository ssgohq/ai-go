package gemini

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// makeIndexedServer returns a test server where each embedding's first value
// encodes its position (index+1)*0.01 — used to verify batch ordering.
func makeIndexedServer(t *testing.T, dimSize int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req geminiBatchEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		resp := geminiBatchEmbedResponse{}
		for i := range req.Requests {
			vec := make([]float32, dimSize)
			for j := range vec {
				vec[j] = float32(i+1) * 0.01
			}
			resp.Embeddings = append(resp.Embeddings, struct {
				Values []float32 `json:"values"`
			}{Values: vec})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

// newIndexedModel creates an EmbeddingModel with a rewriteTransport pointed at server.
func newIndexedModel(t *testing.T, server *httptest.Server) *EmbeddingModel {
	t.Helper()
	return &EmbeddingModel{
		modelID: "text-embedding-004",
		apiKey:  "test-key",
		client: &http.Client{
			Transport: &rewriteTransport{target: server.URL},
		},
	}
}

// TestEmbedBatch_OrderPreserved verifies that the i-th result corresponds
// to the i-th input text after a multi-item batch call.
func TestEmbedBatch_OrderPreserved(t *testing.T) {
	srv := makeIndexedServer(t, 3)
	defer srv.Close()
	m := newIndexedModel(t, srv)

	texts := []string{"alpha", "beta", "gamma", "delta"}
	vecs, err := m.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != len(texts) {
		t.Fatalf("expected %d vectors, got %d", len(texts), len(vecs))
	}
	for i, vec := range vecs {
		want := float32(i+1) * 0.01
		if vec[0] != want {
			t.Errorf("index %d: expected leading value %v (order check), got %v", i, want, vec[0])
		}
	}
}

// TestEmbedBatch_SearchInputs simulates typical embedding calls for cross-conversation
// message search — short query strings that must each receive a full-dimension vector.
func TestEmbedBatch_SearchInputs(t *testing.T) {
	const dim = 768
	srv := makeIndexedServer(t, dim)
	defer srv.Close()
	m := newIndexedModel(t, srv)

	queries := []string{
		"find all messages about project alpha",
		"what did we decide about the database schema",
	}
	vecs, err := m.EmbedBatch(context.Background(), queries)
	if err != nil {
		t.Fatalf("search query embedding: %v", err)
	}
	if len(vecs) != len(queries) {
		t.Fatalf("expected %d vectors, got %d", len(queries), len(vecs))
	}
	for i, vec := range vecs {
		if len(vec) != dim {
			t.Errorf("query[%d]: expected dim=%d, got %d", i, dim, len(vec))
		}
	}
}

// TestEmbedBatch_DocumentChunks simulates the document ingestion pipeline where
// SplitText produces multiple overlapping chunks that are batch-embedded together,
// then stored with their embeddings in the database.
func TestEmbedBatch_DocumentChunks(t *testing.T) {
	const dim = 768
	srv := makeIndexedServer(t, dim)
	defer srv.Close()
	m := newIndexedModel(t, srv)

	chunks := []string{
		"Chapter 1: Introduction to distributed systems and their challenges in modern cloud environments.",
		"Chapter 2: Consensus algorithms such as Raft and Paxos and their trade-offs in practice.",
		"Chapter 3: Replication strategies for high availability including synchronous and asynchronous modes.",
		"Chapter 4: Partition tolerance and the CAP theorem applied to real-world database systems.",
		"Chapter 5: Monitoring and observability for distributed services at scale.",
	}

	vecs, err := m.EmbedBatch(context.Background(), chunks)
	if err != nil {
		t.Fatalf("document chunk embedding: %v", err)
	}
	if len(vecs) != len(chunks) {
		t.Fatalf("expected %d vectors, got %d", len(chunks), len(vecs))
	}
	for i, vec := range vecs {
		if len(vec) == 0 {
			t.Errorf("chunk[%d]: empty embedding", i)
		}
		// Each chunk must map to its own unique embedding (order check via index encoding).
		want := float32(i+1) * 0.01
		if vec[0] != want {
			t.Errorf("chunk[%d]: order mismatch: expected %v, got %v", i, want, vec[0])
		}
	}
}

// TestEmbedBatch_MixedSearchAndChunks simulates the combined search+ingestion
// pattern: search queries and document chunks batched in the same call.
func TestEmbedBatch_MixedSearchAndChunks(t *testing.T) {
	const dim = 768
	srv := makeIndexedServer(t, dim)
	defer srv.Close()
	m := newIndexedModel(t, srv)

	searchInputs := []string{
		"find all messages about project alpha",
		"what did we decide about the database schema",
	}
	docChunks := []string{
		"Introduction to distributed systems...",
		"Consensus algorithms and their trade-offs...",
		"Replication strategies for high availability...",
	}
	all := append(searchInputs, docChunks...)

	vecs, err := m.EmbedBatch(context.Background(), all)
	if err != nil {
		t.Fatalf("mixed batch embedding: %v", err)
	}
	if len(vecs) != len(all) {
		t.Fatalf("expected %d vectors, got %d", len(all), len(vecs))
	}
	for i, vec := range vecs {
		if len(vec) != dim {
			t.Errorf("item[%d]: expected dim=%d, got %d", i, dim, len(vec))
		}
	}
}

// TestEmbed_DelegatesToBatch verifies that Embed is a thin wrapper over EmbedBatch.
func TestEmbed_DelegatesToBatch(t *testing.T) {
	srv := makeIndexedServer(t, 4)
	defer srv.Close()
	m := newIndexedModel(t, srv)

	vec, err := m.Embed(context.Background(), "single text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vec) != 4 {
		t.Errorf("expected 4-dim vector, got %d", len(vec))
	}
	// Index-encoding: single item = index 0, so value = 1*0.01.
	if vec[0] != 0.01 {
		t.Errorf("expected vec[0]=0.01, got %v", vec[0])
	}
}

// TestBuildBatchEmbedRequest_ModelPrefix verifies that model IDs are prefixed
// with "models/" as required by the Gemini batchEmbedContents API.
func TestBuildBatchEmbedRequest_ModelPrefix(t *testing.T) {
	data, err := buildBatchEmbedRequest("text-embedding-004", []string{"x"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var req geminiBatchEmbedRequest
	if err := json.Unmarshal(data, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.Requests[0].Model != "models/text-embedding-004" {
		t.Errorf("expected model prefix, got %q", req.Requests[0].Model)
	}
}

// TestParseBatchEmbedResponse_Valid verifies correct extraction of float vectors.
func TestParseBatchEmbedResponse_Valid(t *testing.T) {
	resp := geminiBatchEmbedResponse{
		Embeddings: []struct {
			Values []float32 `json:"values"`
		}{
			{Values: []float32{0.1, 0.2}},
			{Values: []float32{0.3, 0.4}},
		},
	}
	body, _ := json.Marshal(resp)
	results, err := parseBatchEmbedResponse(body, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if results[0][1] != 0.2 {
		t.Errorf("expected 0.2, got %v", results[0][1])
	}
	if results[1][0] != 0.3 {
		t.Errorf("expected 0.3, got %v", results[1][0])
	}
}
