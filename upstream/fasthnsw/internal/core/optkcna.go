package core

import (
	"fmt"
)

// optKCNAConfig controls one OptKCNA candidate-refresh step.
type optKCNAConfig struct {
	CandidateK   int
	SearchEf     int
	MaxDegree    int
	AlphaDegrees float64
	Workers      int
}

// optKCNA refreshes k-CNA candidates by searching an intermediate alpha-PG.
//
// The current candidate lists are first converted to an alpha-pruned graph.
// FastHNSW does not run NSG-style DFS connectivity enhancement on this
// temporary graph; Section 5.3 describes HNSW layers as RNG indexes built by
// pruning k-CNA results without that repair step. Each node searches the graph
// using its own vector as the query, drops itself from the results, and keeps
// up to CandidateK nearest candidates for the next construction iteration.
func optKCNA(candidates [][]candidate, cfg optKCNAConfig, vectors []float32, dim int, metric Metric) ([][]candidate, error) {
	count, err := validateOptKCNAInput(candidates, cfg, vectors, dim, metric)
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return [][]candidate{}, nil
	}

	alphaGraph, err := buildPrunedLayer(candidates, cfg.MaxDegree, alphaPruneMode(cfg.AlphaDegrees), vectors, dim, metric, cfg.Workers)
	if err != nil {
		return nil, err
	}

	refreshed := make([][]candidate, count)
	workerCount := effectiveWorkerCount(cfg.Workers, count)
	scratches := makeGraphSearchScratches(workerCount, count, cfg.SearchEf)
	_ = parallelForNodes(count, cfg.Workers, func(workerID int, sourceID int) error {
		results := graphSearchLayer(alphaGraph, vectors, dim, metric, sourceID, vectorAt(vectors, dim, sourceID), cfg.SearchEf, &scratches[workerID])
		refreshed[sourceID] = candidatesFromResults(sourceID, results, cfg.CandidateK)
		return nil
	})
	return refreshed, nil
}

func validateOptKCNAInput(candidates [][]candidate, cfg optKCNAConfig, vectors []float32, dim int, metric Metric) (int, error) {
	if cfg.CandidateK <= 0 {
		return 0, fmt.Errorf("fasthnsw: CandidateK must be positive")
	}
	if cfg.SearchEf < cfg.CandidateK {
		return 0, fmt.Errorf("fasthnsw: SearchEf must be greater than or equal to CandidateK")
	}
	if cfg.MaxDegree <= 0 {
		return 0, fmt.Errorf("fasthnsw: MaxDegree must be positive")
	}
	if cfg.AlphaDegrees < minAlpha || cfg.AlphaDegrees > maxAlphaDegrees {
		return 0, fmt.Errorf("fasthnsw: AlphaDegrees must be in [%.0f, %.0f]", float64(minAlpha), float64(maxAlphaDegrees))
	}
	if cfg.Workers < 0 {
		return 0, fmt.Errorf("fasthnsw: Workers must be positive")
	}
	if !validMetric(metric) {
		return 0, fmt.Errorf("fasthnsw: unsupported metric %d", metric)
	}
	if dim <= 0 {
		return 0, fmt.Errorf("fasthnsw: vector dimension must be positive")
	}
	if len(vectors)%dim != 0 {
		return 0, fmt.Errorf("fasthnsw: flat vector storage is not aligned to dimension")
	}

	count := len(vectors) / dim
	if len(candidates) != count {
		return 0, fmt.Errorf("fasthnsw: candidate list count %d, want %d", len(candidates), count)
	}
	return count, nil
}

func candidatesFromResults(sourceID int, results []Result, limit int) []candidate {
	out := make([]candidate, 0, limit)
	for _, result := range results {
		if result.ID == sourceID {
			continue
		}
		out = append(out, candidate{id: result.ID, distance: result.Distance})
		if len(out) >= limit {
			break
		}
	}
	return out
}
