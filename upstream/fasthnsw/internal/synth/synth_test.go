package synth

import (
	"math"
	"strconv"
	"testing"
)

func TestGeneratedDatasetsAreDeterministic(t *testing.T) {
	if !SameVectors(UniformVectors(16, 3), UniformVectors(16, 3)) {
		t.Fatal("UniformVectors is not deterministic")
	}
	if !SameVectors(UniformQueries(16, 3), UniformQueries(16, 3)) {
		t.Fatal("UniformQueries is not deterministic")
	}
	if !SameVectors(ClusteredVectors(24, 4, 3), ClusteredVectors(24, 4, 3)) {
		t.Fatal("ClusteredVectors is not deterministic")
	}
	if !SameVectors(ClusteredQueries(12, 4, 3), ClusteredQueries(12, 4, 3)) {
		t.Fatal("ClusteredQueries is not deterministic")
	}
}

func TestClusteredVectorsAvoidExactDuplicateCycles(t *testing.T) {
	vectors := ClusteredVectors(1000, 16, 6)
	seen := make(map[string]int, len(vectors))
	for id, vector := range vectors {
		key := vectorKey(vector)
		if previousID, ok := seen[key]; ok {
			t.Fatalf("ClusteredVectors produced duplicate vectors at ids %d and %d", previousID, id)
		}
		seen[key] = id
	}
}

func vectorKey(vector []float32) string {
	key := make([]byte, 0, len(vector)*12)
	for _, value := range vector {
		key = strconv.AppendUint(key, uint64(math.Float32bits(value)), 10)
		key = append(key, ',')
	}
	return string(key)
}
