package core

import (
	"fmt"
	"io"
)

// Index stores vectors and graph metadata for approximate nearest-neighbor
// search. Its fields are intentionally private to keep the public API stable.
type Index struct {
	cfg Config

	// vectors is a flat count*dim block, even though Build accepts [][]float32.
	// The flat layout improves locality for distance-heavy construction and
	// search, avoids retaining caller-owned slice backing arrays, and keeps the
	// future persistence format straightforward.
	vectors []float32
	dim     int
	count   int

	// layers stores HNSW adjacency as layers[layer][nodeID] -> neighbor ids.
	// Build populates this metadata through global layer-by-layer construction;
	// tests may still install hand-built graphs to isolate search behavior.
	layers     [][][]int
	levels     []int
	entryPoint int
	maxLayer   int
	graphReady bool
}

// New creates an index with validated configuration.
func New(cfg Config) (*Index, error) {
	cfg, err := normalizeConfig(cfg)
	if err != nil {
		return nil, err
	}
	return &Index{cfg: cfg, entryPoint: -1, maxLayer: -1}, nil
}

// Build validates a dataset, copies it into the index-owned flat vector store,
// and constructs the searchable FastHNSW graph.
func (idx *Index) Build(vectors [][]float32) error {
	if idx == nil {
		return fmt.Errorf("fasthnsw: nil index")
	}
	flat, dim, err := FlattenVectors(vectors, idx.cfg.Dim, idx.cfg.Metric)
	if err != nil {
		return err
	}
	idx.vectors = flat
	idx.dim = dim
	idx.count = len(vectors)
	idx.cfg.Dim = dim
	idx.resetSearchableGraph()
	if err := idx.buildSearchableGraph(); err != nil {
		idx.resetSearchableGraph()
		return err
	}
	return nil
}

// resetSearchableGraph clears graph metadata when index-owned vectors are
// replaced or future construction rebuilds the graph. It only resets state;
// graph invariants are established by construction code, not revalidated here.
func (idx *Index) resetSearchableGraph() {
	idx.layers = nil
	idx.levels = nil
	idx.entryPoint = -1
	idx.maxLayer = -1
	idx.graphReady = false
}

// Search returns the approximate nearest neighbors for query using HNSW graph
// traversal over the graph produced by Build.
func (idx *Index) Search(query []float32, k int, efSearch int) ([]Result, error) {
	if idx == nil {
		return nil, fmt.Errorf("fasthnsw: nil index")
	}
	if k <= 0 {
		return nil, fmt.Errorf("fasthnsw: k must be positive")
	}
	if efSearch <= 0 {
		return nil, fmt.Errorf("fasthnsw: efSearch must be positive")
	}
	if efSearch < k {
		return nil, fmt.Errorf("fasthnsw: efSearch must be greater than or equal to k")
	}
	dim := idx.cfg.Dim
	if idx.dim > 0 {
		dim = idx.dim
	}
	if err := validateVector(query, dim, "query"); err != nil {
		return nil, err
	}
	preparedQuery, err := prepareQuery(idx.cfg.Metric, query)
	if err != nil {
		return nil, err
	}
	if !idx.graphReady {
		return nil, ErrIndexNotBuilt
	}
	return idx.search(preparedQuery, k, efSearch)
}

// Save writes a versioned binary representation of the built index.
func (idx *Index) Save(w io.Writer) error {
	if idx == nil {
		return fmt.Errorf("fasthnsw: nil index")
	}
	if w == nil {
		return fmt.Errorf("fasthnsw: nil writer")
	}
	return idx.save(w)
}

// Load reads a versioned binary index representation written by Save.
func Load(r io.Reader) (*Index, error) {
	if r == nil {
		return nil, fmt.Errorf("fasthnsw: nil reader")
	}
	return loadIndex(r)
}

// validateVectors checks the public dataset shape and returns its dimension.
func validateVectors(vectors [][]float32, configuredDim int) (int, error) {
	if len(vectors) == 0 {
		return 0, fmt.Errorf("fasthnsw: vectors must not be empty")
	}

	dim := configuredDim
	if dim == 0 {
		dim = len(vectors[0])
	}
	if dim <= 0 {
		return 0, fmt.Errorf("fasthnsw: vector dimension must be positive")
	}

	for i, vector := range vectors {
		if len(vector) != dim {
			return 0, fmt.Errorf("fasthnsw: vector %d has dimension %d, want %d", i, len(vector), dim)
		}
	}
	return dim, nil
}

// validateVector checks one public query or vector against a configured
// dimension. A zero configured dimension means the caller has not fixed a
// dimension yet.
func validateVector(vector []float32, configuredDim int, name string) error {
	if len(vector) == 0 {
		return fmt.Errorf("fasthnsw: %s vector must not be empty", name)
	}
	if configuredDim > 0 && len(vector) != configuredDim {
		return fmt.Errorf("fasthnsw: %s vector has dimension %d, want %d", name, len(vector), configuredDim)
	}
	return nil
}
