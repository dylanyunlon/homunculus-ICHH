package core

import (
	"strings"
	"testing"
)

func TestBuildPrunedLayerFromExactCandidates(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{0, 1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates, err := exactCandidates(flat, dim, MetricL2, 2, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}

	got, err := buildPrunedLayer(candidates, 2, rngPruneMode(), flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildPrunedLayer returned error: %v", err)
	}
	want := [][]int{
		{1, 2},
		{0},
		{0},
	}
	assertAdjacency(t, got, want)
}

func TestBuildPrunedLayerAddsReverseEdges(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{10},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := [][]candidate{
		{{id: 1}},
		nil,
		nil,
	}

	got, err := buildPrunedLayer(candidates, 2, rngPruneMode(), flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildPrunedLayer returned error: %v", err)
	}
	want := [][]int{
		{1},
		{0},
		nil,
	}
	assertAdjacency(t, got, want)
}

func TestBuildPrunedLayerNormalizesSelfAndDuplicates(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := [][]candidate{
		{{id: 0}, {id: 1}, {id: 1}, {id: 2}},
		{{id: 1}, {id: 0}},
		{{id: 2}, {id: 0}},
	}

	got, err := buildPrunedLayer(candidates, 2, rngPruneMode(), flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildPrunedLayer returned error: %v", err)
	}
	for sourceID, neighbors := range got {
		seen := make(map[int]bool, len(neighbors))
		for _, neighborID := range neighbors {
			if neighborID == sourceID {
				t.Fatalf("adjacency[%d] contains self edge: %v", sourceID, neighbors)
			}
			if seen[neighborID] {
				t.Fatalf("adjacency[%d] contains duplicate edge %d: %v", sourceID, neighborID, neighbors)
			}
			seen[neighborID] = true
		}
	}
}

func TestBuildPrunedLayerEnforcesMaxDegreeAfterReverseMerge(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{-1, 0},
		{0, 1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := [][]candidate{
		nil,
		{{id: 0}},
		{{id: 0}},
		{{id: 0}},
	}

	got, err := buildPrunedLayer(candidates, 1, rngPruneMode(), flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildPrunedLayer returned error: %v", err)
	}
	for sourceID, neighbors := range got {
		if len(neighbors) > 1 {
			t.Fatalf("len(adjacency[%d]) = %d, want <= 1: %v", sourceID, len(neighbors), neighbors)
		}
	}
}

func TestBuildPrunedLayerAlphaModeCanRetainRNGPrunedEdge(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{1, 1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := [][]candidate{
		{{id: 1}, {id: 2}},
		nil,
		nil,
	}

	rng, err := buildPrunedLayer(candidates, 4, rngPruneMode(), flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildPrunedLayer RNG returned error: %v", err)
	}
	alpha, err := buildPrunedLayer(candidates, 4, alphaPruneMode(120), flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildPrunedLayer alpha returned error: %v", err)
	}

	assertAdjacency(t, [][]int{rng[0]}, [][]int{{1}})
	assertAdjacency(t, [][]int{alpha[0]}, [][]int{{1, 2}})
}

func TestBuildPrunedLayerRejectsBadCandidateID(t *testing.T) {
	_, err := buildPrunedLayer([][]candidate{{{id: 2}}, nil}, 1, rngPruneMode(), []float32{0, 1}, 1, MetricL2, 1)
	if err == nil {
		t.Fatal("buildPrunedLayer returned nil error")
	}
}

func TestBuildPrunedLayerParallelReturnsFirstSourceError(t *testing.T) {
	_, err := buildPrunedLayer([][]candidate{
		{{id: 9}},
		{{id: 8}},
	}, 1, rngPruneMode(), []float32{0, 1}, 1, MetricL2, 2)
	if err == nil {
		t.Fatal("buildPrunedLayer returned nil error")
	}
	if got, want := err.Error(), "candidate id 9"; !strings.Contains(got, want) {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func assertAdjacency(t *testing.T, got, want [][]int) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(adjacency) = %d, want %d: %v", len(got), len(want), got)
	}
	for sourceID := range want {
		if len(got[sourceID]) != len(want[sourceID]) {
			t.Fatalf("len(adjacency[%d]) = %d, want %d: %v", sourceID, len(got[sourceID]), len(want[sourceID]), got[sourceID])
		}
		for i := range want[sourceID] {
			if got[sourceID][i] != want[sourceID][i] {
				t.Fatalf("adjacency[%d][%d] = %d, want %d", sourceID, i, got[sourceID][i], want[sourceID][i])
			}
		}
	}
}
