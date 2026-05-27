package benchdata

import (
	"fmt"
	"io"
	"os"
	"time"

	"github.com/cryo-zd/fasthnsw"
	"github.com/cryo-zd/fasthnsw/internal/core"
	"github.com/cryo-zd/fasthnsw/internal/eval"
)

// ValidationAlgorithm selects the construction path used by validation. It is
// internal to benchmark tooling; the public package continues to expose only
// the FastHNSW Build method.
type ValidationAlgorithm string

const (
	// AlgorithmFastHNSW uses the library's public FastHNSW construction path.
	AlgorithmFastHNSW ValidationAlgorithm = "fasthnsw"
	// AlgorithmHNSW uses the internal standard incremental HNSW baseline.
	AlgorithmHNSW ValidationAlgorithm = "hnsw"
)

// Dataset is an in-memory benchmark dataset. It is intentionally internal so
// public users do not inherit benchmark-loader compatibility constraints.
type Dataset struct {
	Name        string
	Metric      fasthnsw.Metric
	Base        [][]float32
	Queries     [][]float32
	GroundTruth [][]int
}

// Dim returns the dataset vector dimension.
func (d Dataset) Dim() int {
	if len(d.Base) == 0 {
		return 0
	}
	return len(d.Base[0])
}

// Validate checks that the dataset can be used for Recall@k validation.
func (d Dataset) Validate(k int) error {
	if d.Name == "" {
		return fmt.Errorf("dataset name must not be empty")
	}
	if len(d.Base) == 0 {
		return fmt.Errorf("dataset base vectors must not be empty")
	}
	if len(d.Queries) == 0 {
		return fmt.Errorf("dataset queries must not be empty")
	}
	dim := len(d.Base[0])
	if dim == 0 {
		return fmt.Errorf("dataset dimension must be positive")
	}
	for i, vector := range d.Base {
		if len(vector) != dim {
			return fmt.Errorf("base vector %d has dimension %d, want %d", i, len(vector), dim)
		}
	}
	for i, query := range d.Queries {
		if len(query) != dim {
			return fmt.Errorf("query vector %d has dimension %d, want %d", i, len(query), dim)
		}
	}
	if len(d.GroundTruth) != len(d.Queries) {
		return fmt.Errorf("ground truth count %d does not match query count %d", len(d.GroundTruth), len(d.Queries))
	}
	for i, neighbors := range d.GroundTruth {
		if len(neighbors) < k {
			return fmt.Errorf("ground truth row %d has %d neighbors, want at least %d", i, len(neighbors), k)
		}
		for rank := 0; rank < k; rank++ {
			if neighbors[rank] < 0 || neighbors[rank] >= len(d.Base) {
				return fmt.Errorf("ground truth row %d rank %d has id %d outside loaded base size %d", i, rank, neighbors[rank], len(d.Base))
			}
		}
	}
	return nil
}

// ValidationResult is the machine-readable summary produced by a validation
// run over one dataset and one build/search configuration.
type ValidationResult struct {
	Algorithm   ValidationAlgorithm
	Dataset     Dataset
	K           int
	EfSearch    int
	BuildTime   time.Duration
	QueryTime   time.Duration
	Recall      float64
	IndexBytes  int64
	SearchCount int
}

// QPS returns single-query throughput over the measured search loop.
func (r ValidationResult) QPS() float64 {
	if r.QueryTime <= 0 {
		return 0
	}
	return float64(r.SearchCount) / r.QueryTime.Seconds()
}

// RunValidation builds an index, searches every query, and compares results
// against the dataset ground truth. If saveIndexPath is non-empty, the built
// index is persisted and the resulting file size is reported.
func RunValidation(dataset Dataset, cfg fasthnsw.Config, k int, efSearch int, saveIndexPath string) (ValidationResult, error) {
	return RunValidationWithAlgorithm(dataset, cfg, AlgorithmFastHNSW, k, efSearch, saveIndexPath)
}

// RunValidationWithAlgorithm is RunValidation with an explicit internal
// construction algorithm selection for CLI and benchmark comparisons.
func RunValidationWithAlgorithm(dataset Dataset, cfg fasthnsw.Config, algorithm ValidationAlgorithm, k int, efSearch int, saveIndexPath string) (ValidationResult, error) {
	if k <= 0 {
		return ValidationResult{}, fmt.Errorf("k must be positive")
	}
	if efSearch < k {
		return ValidationResult{}, fmt.Errorf("ef must be greater than or equal to k")
	}
	if err := dataset.Validate(k); err != nil {
		return ValidationResult{}, err
	}
	if err := validateAlgorithm(algorithm); err != nil {
		return ValidationResult{}, err
	}

	cfg.Metric = dataset.Metric
	if cfg.Dim == 0 {
		cfg.Dim = dataset.Dim()
	}

	buildStart := time.Now()
	idx, err := buildValidationIndex(dataset.Base, cfg, algorithm)
	if err != nil {
		return ValidationResult{}, err
	}
	buildTime := time.Since(buildStart)

	var indexBytes int64
	if saveIndexPath != "" {
		file, err := os.Create(saveIndexPath)
		if err != nil {
			return ValidationResult{}, fmt.Errorf("create saved index: %w", err)
		}
		if err := idx.Save(file); err != nil {
			if closeErr := file.Close(); closeErr != nil {
				return ValidationResult{}, fmt.Errorf("save index: %w; close index: %w", err, closeErr)
			}
			return ValidationResult{}, fmt.Errorf("save index: %w", err)
		}
		if err := file.Close(); err != nil {
			return ValidationResult{}, fmt.Errorf("close saved index: %w", err)
		}
		info, err := os.Stat(saveIndexPath)
		if err != nil {
			return ValidationResult{}, fmt.Errorf("stat saved index: %w", err)
		}
		indexBytes = info.Size()
	}

	searchStart := time.Now()
	var recallTotal float64
	for queryID, query := range dataset.Queries {
		results, err := idx.Search(query, k, efSearch)
		if err != nil {
			return ValidationResult{}, fmt.Errorf("query %d: %w", queryID, err)
		}
		recallTotal += eval.RecallAtK(core.ResultIDs(results, k), dataset.GroundTruth[queryID], k)
	}
	queryTime := time.Since(searchStart)

	return ValidationResult{
		Algorithm:   algorithm,
		Dataset:     dataset,
		K:           k,
		EfSearch:    efSearch,
		BuildTime:   buildTime,
		QueryTime:   queryTime,
		Recall:      recallTotal / float64(len(dataset.Queries)),
		IndexBytes:  indexBytes,
		SearchCount: len(dataset.Queries),
	}, nil
}

type validationIndex interface {
	Search(query []float32, k int, efSearch int) ([]fasthnsw.Result, error)
	Save(w io.Writer) error
}

func validateAlgorithm(algorithm ValidationAlgorithm) error {
	switch algorithm {
	case AlgorithmFastHNSW, AlgorithmHNSW:
		return nil
	default:
		return fmt.Errorf("unsupported validation algorithm %q", algorithm)
	}
}

func buildValidationIndex(vectors [][]float32, cfg fasthnsw.Config, algorithm ValidationAlgorithm) (validationIndex, error) {
	switch algorithm {
	case AlgorithmFastHNSW:
		idx, err := fasthnsw.New(cfg)
		if err != nil {
			return nil, err
		}
		if err := idx.Build(vectors); err != nil {
			return nil, err
		}
		return idx, nil
	case AlgorithmHNSW:
		return core.BuildStandardHNSWForBenchmark(core.Config(cfg), vectors)
	default:
		return nil, fmt.Errorf("unsupported validation algorithm %q", algorithm)
	}
}

// ExactGroundTruth computes deterministic exact nearest-neighbor ids for
// generated validation datasets that do not ship precomputed ground truth.
func ExactGroundTruth(base [][]float32, queries [][]float32, metric fasthnsw.Metric, k int) ([][]int, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}
	if len(base) == 0 || len(queries) == 0 {
		return nil, fmt.Errorf("base and query vectors must not be empty")
	}
	if k > len(base) {
		k = len(base)
	}
	flatBase, dim, err := core.FlattenVectors(base, 0, metric)
	if err != nil {
		return nil, err
	}

	truth := make([][]int, len(queries))
	for queryID, query := range queries {
		results, err := core.ExactTopK(flatBase, dim, metric, query, k)
		if err != nil {
			return nil, fmt.Errorf("query %d: %w", queryID, err)
		}
		ids := make([]int, len(results))
		for i, result := range results {
			ids[i] = result.ID
		}
		truth[queryID] = ids
	}
	return truth, nil
}
