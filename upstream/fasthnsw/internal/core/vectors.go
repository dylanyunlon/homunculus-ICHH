package core

import "fmt"

// FlattenVectors validates and copies public [][]float32 input into the flat
// vector storage used internally by Index.
//
// The public API accepts [][]float32 because it is convenient for callers, but
// the index owns one contiguous []float32 block internally. That layout keeps
// distance-heavy construction and search more cache-friendly, avoids retaining
// user-owned slice backing arrays, reduces per-vector slice header and GC
// overhead, and makes future persistence a simple count*dim float block. For
// cosine indexes, this is also where the stored representation is normalized
// without mutating caller input.
func FlattenVectors(vectors [][]float32, configuredDim int, metric Metric) ([]float32, int, error) {
	dim, err := validateVectors(vectors, configuredDim)
	if err != nil {
		return nil, 0, err
	}

	flat := make([]float32, len(vectors)*dim)
	for i, vector := range vectors {
		dst := flat[i*dim : (i+1)*dim]
		switch metric {
		case MetricL2:
			copy(dst, vector)
		case MetricCosine:
			if err := normalizeInto(dst, vector); err != nil {
				return nil, 0, fmt.Errorf("fasthnsw: vector %d: %w", i, err)
			}
		default:
			return nil, 0, fmt.Errorf("fasthnsw: unsupported metric %d", metric)
		}
	}
	return flat, dim, nil
}

// vectorAt returns a view of one vector in flat storage.
// Callers must pass a valid id; this helper is kept small for hot distance
// loops and relies on earlier validation and graph bounds checks.
func vectorAt(vectors []float32, dim int, id int) []float32 {
	start := id * dim
	return vectors[start : start+dim]
}
