package core

import "testing"

func TestExactTopKL2(t *testing.T) {
	vectors := []float32{
		0, 0,
		2, 0,
		1, 0,
	}

	got, err := ExactTopK(vectors, 2, MetricL2, []float32{0.9, 0}, 2)
	if err != nil {
		t.Fatalf("ExactTopK returned error: %v", err)
	}

	want := []Result{
		{ID: 2, Distance: 0.010000004},
		{ID: 0, Distance: 0.80999994},
	}
	assertResults(t, got, want)
}

func TestExactTopKTieBreaksByID(t *testing.T) {
	vectors := []float32{
		1, 0,
		-1, 0,
	}

	got, err := ExactTopK(vectors, 2, MetricL2, []float32{0, 0}, 2)
	if err != nil {
		t.Fatalf("ExactTopK returned error: %v", err)
	}

	want := []Result{
		{ID: 0, Distance: 1},
		{ID: 1, Distance: 1},
	}
	assertResults(t, got, want)
}

func TestExactTopKCosineNormalizesQuery(t *testing.T) {
	vectors, dim, err := FlattenVectors([][]float32{{1, 0}, {0, 1}}, 0, MetricCosine)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := ExactTopK(vectors, dim, MetricCosine, []float32{10, 0}, 2)
	if err != nil {
		t.Fatalf("ExactTopK returned error: %v", err)
	}

	want := []Result{
		{ID: 0, Distance: 0},
		{ID: 1, Distance: 1},
	}
	assertResults(t, got, want)
}

func TestExactTopKRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name    string
		vectors []float32
		dim     int
		query   []float32
		k       int
	}{
		{name: "bad dim", vectors: []float32{1, 2}, dim: 0, query: []float32{1}, k: 1},
		{name: "unaligned", vectors: []float32{1, 2, 3}, dim: 2, query: []float32{1, 2}, k: 1},
		{name: "bad query dim", vectors: []float32{1, 2}, dim: 2, query: []float32{1}, k: 1},
		{name: "bad k", vectors: []float32{1, 2}, dim: 2, query: []float32{1, 2}, k: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ExactTopK(tt.vectors, tt.dim, MetricL2, tt.query, tt.k); err == nil {
				t.Fatal("ExactTopK returned nil error")
			}
		})
	}
}
