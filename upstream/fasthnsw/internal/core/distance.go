package core

import (
	"fmt"
	"math"
)

// squaredL2Generic returns squared Euclidean distance without taking a square
// root. ANN ranking only needs relative ordering, so avoiding sqrt keeps the
// hot distance loop cheaper. Platform wrappers may replace squaredL2 with a
// SIMD implementation, but this generic kernel remains the correctness
// fallback on every build.
func squaredL2Generic(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// dotGeneric returns the dot product for equal-length vectors. Platform
// wrappers may replace dot with a SIMD implementation.
func dotGeneric(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// normalizeInto writes a unit-length copy of src into dst.
// Cosine distance is evaluated over normalized vectors, and zero vectors are
// rejected because their cosine similarity is undefined.
func normalizeInto(dst, src []float32) error {
	var normSquared float64
	for _, value := range src {
		v := float64(value)
		normSquared += v * v
	}
	if normSquared == 0 {
		return fmt.Errorf("fasthnsw: cosine vectors must not be zero vectors")
	}

	invNorm := float32(1 / math.Sqrt(normSquared))
	for i, value := range src {
		dst[i] = value * invNorm
	}
	return nil
}

// cosineDistanceNormalized returns 1 - dot(a, b) and assumes both inputs have
// already been normalized to unit length.
func cosineDistanceNormalized(a, b []float32) float32 {
	return 1 - dot(a, b)
}

// distance dispatches to the metric-specific distance implementation.
func distance(metric Metric, a, b []float32) float32 {
	switch metric {
	case MetricL2:
		return squaredL2(a, b)
	case MetricCosine:
		return cosineDistanceNormalized(a, b)
	default:
		panic("fasthnsw: unsupported metric reached distance")
	}
}

// prepareQuery returns the query representation expected by distance.
// L2 queries are used directly, while cosine queries are normalized into a new
// slice so callers never observe a mutation of their input.
func prepareQuery(metric Metric, query []float32) ([]float32, error) {
	if metric != MetricCosine {
		return query, nil
	}

	normalized := make([]float32, len(query))
	if err := normalizeInto(normalized, query); err != nil {
		return nil, err
	}
	return normalized, nil
}
