package core

import (
	"math"
	"testing"
)

func almostEqual(got, want float32) bool {
	return math.Abs(float64(got-want)) < 1e-6
}

func assertResults(t *testing.T, got, want []Result) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(results) = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("result %d ID = %d, want %d", i, got[i].ID, want[i].ID)
		}
		if !almostEqual(got[i].Distance, want[i].Distance) {
			t.Fatalf("result %d distance = %v, want %v", i, got[i].Distance, want[i].Distance)
		}
	}
}

func mustBuildIndex(t *testing.T, cfg Config, vectors [][]float32) *Index {
	t.Helper()

	idx, err := New(cfg)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	return idx
}

// lineVectors returns one-dimensional points on a line. These are geometric
// fixtures for pruning and graph-construction tests, not general synthetic
// benchmark data.
func lineVectors(count int) [][]float32 {
	vectors := make([][]float32, count)
	for id := 0; id < count; id++ {
		vectors[id] = []float32{float32(id)}
	}
	return vectors
}
