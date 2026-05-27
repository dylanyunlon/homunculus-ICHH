package core

import (
	"reflect"
	"testing"

	"github.com/cryo-zd/fasthnsw/internal/eval"
	"github.com/cryo-zd/fasthnsw/internal/synth"
)

func TestBuildStandardHNSWSearchesTinyDataset(t *testing.T) {
	vectors := lineVectors(6)
	idx, err := BuildStandardHNSWForBenchmark(Config{Dim: 1, M: 4, ConstructionL: 8, Seed: 7}, vectors)
	if err != nil {
		t.Fatalf("BuildStandardHNSWForBenchmark returned error: %v", err)
	}

	query := []float32{2.2}
	got, err := idx.Search(query, 3, 8)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	want, err := ExactTopK(idx.vectors, idx.dim, idx.cfg.Metric, query, 3)
	if err != nil {
		t.Fatalf("ExactTopK returned error: %v", err)
	}
	assertResults(t, got, want)
}

func TestStandardHNSWDegreeBounds(t *testing.T) {
	idx, err := BuildStandardHNSWForBenchmark(Config{Dim: 4, M: 4, ConstructionL: 12, Seed: 13}, synth.UniformVectors(160, 4))
	if err != nil {
		t.Fatalf("BuildStandardHNSWForBenchmark returned error: %v", err)
	}

	for layer, adjacency := range idx.layers {
		maxDegree := standardLayerMaxDegree(idx.cfg, layer)
		for nodeID, neighbors := range adjacency {
			if len(neighbors) > maxDegree {
				t.Fatalf("layer %d node %d degree = %d, want <= %d", layer, nodeID, len(neighbors), maxDegree)
			}
		}
	}
}

func TestStandardHNSWBuildIsDeterministic(t *testing.T) {
	vectors := synth.ClusteredVectors(96, 4, 8)
	cfg := Config{Dim: 4, M: 4, ConstructionL: 12, Seed: 19}

	left, err := BuildStandardHNSWForBenchmark(cfg, vectors)
	if err != nil {
		t.Fatalf("left BuildStandardHNSWForBenchmark returned error: %v", err)
	}
	right, err := BuildStandardHNSWForBenchmark(cfg, vectors)
	if err != nil {
		t.Fatalf("right BuildStandardHNSWForBenchmark returned error: %v", err)
	}

	if !reflect.DeepEqual(left.levels, right.levels) {
		t.Fatal("standard HNSW levels differ under fixed seed")
	}
	if left.entryPoint != right.entryPoint || left.maxLayer != right.maxLayer {
		t.Fatalf("standard HNSW graph metadata differs: entry %d/%d maxLayer %d/%d", left.entryPoint, right.entryPoint, left.maxLayer, right.maxLayer)
	}
	if !reflect.DeepEqual(left.layers, right.layers) {
		t.Fatal("standard HNSW layers differ under fixed seed")
	}
}

func TestSelectStandardHNSWNeighborsUsesDiversityHeuristic(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{2},
		{-1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := []Result{
		resultForNode(flat, dim, MetricL2, 1, vectorAt(flat, dim, 0)),
		resultForNode(flat, dim, MetricL2, 2, vectorAt(flat, dim, 0)),
		resultForNode(flat, dim, MetricL2, 3, vectorAt(flat, dim, 0)),
	}

	got, err := selectStandardHNSWNeighbors(0, candidates, 3, make([][]int, 4), flat, dim, MetricL2, false, false)
	if err != nil {
		t.Fatalf("selectStandardHNSWNeighbors returned error: %v", err)
	}
	assertIntSlice(t, got, []int{1, 3})

	got, err = selectStandardHNSWNeighbors(0, candidates, 3, make([][]int, 4), flat, dim, MetricL2, false, true)
	if err != nil {
		t.Fatalf("selectStandardHNSWNeighbors keep-pruned returned error: %v", err)
	}
	assertIntSlice(t, got, []int{1, 3, 2})
}

func TestStandardHNSWRecallOnGeneratedData(t *testing.T) {
	vectors := synth.ClusteredVectors(240, 8, 12)
	queries := synth.ClusteredQueries(24, 8, 12)
	idx, err := BuildStandardHNSWForBenchmark(Config{Dim: 8, M: 8, ConstructionL: 24, Seed: 23}, vectors)
	if err != nil {
		t.Fatalf("BuildStandardHNSWForBenchmark returned error: %v", err)
	}

	var hits int
	var total int
	for _, query := range queries {
		exact, err := ExactTopK(idx.vectors, idx.dim, idx.cfg.Metric, query, 10)
		if err != nil {
			t.Fatalf("ExactTopK returned error: %v", err)
		}
		got, err := idx.Search(query, 10, 32)
		if err != nil {
			t.Fatalf("Search returned error: %v", err)
		}
		queryHits, queryTotal := eval.CountHitsAtK(ResultIDs(got, 10), ResultIDs(exact, 10), 10)
		hits += queryHits
		total += queryTotal
	}
	recall := float64(hits) / float64(total)
	if recall < 0.85 {
		t.Fatalf("standard HNSW Recall@10 = %.3f, want >= 0.85", recall)
	}
}
