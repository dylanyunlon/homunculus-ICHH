package core

import (
	"testing"

	"github.com/cryo-zd/fasthnsw/internal/eval"
)

func TestExactCandidates(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{3, 0},
		{0, 2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := exactCandidates(flat, dim, MetricL2, 2, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}

	want := [][]candidate{
		{{id: 1, distance: 1}, {id: 3, distance: 4}},
		{{id: 0, distance: 1}, {id: 2, distance: 4}},
		{{id: 1, distance: 4}, {id: 0, distance: 9}},
		{{id: 0, distance: 4}, {id: 1, distance: 5}},
	}
	assertCandidates(t, got, want)
}

func TestExactCandidatesExcludesSelfAndTruncatesK(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := exactCandidates(flat, dim, MetricL2, 10, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("len(got) = %d, want 3", len(got))
	}
	for sourceID, candidates := range got {
		if len(candidates) != 2 {
			t.Fatalf("len(got[%d]) = %d, want 2", sourceID, len(candidates))
		}
		for _, candidate := range candidates {
			if candidate.id == sourceID {
				t.Fatalf("got[%d] contains self candidate: %+v", sourceID, candidates)
			}
		}
	}
}

func TestExactCandidatesTieBreaksByID(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{-1},
		{1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := exactCandidates(flat, dim, MetricL2, 2, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}
	want := [][]candidate{
		{{id: 1, distance: 1}, {id: 2, distance: 1}},
		{{id: 0, distance: 1}, {id: 2, distance: 4}},
		{{id: 0, distance: 1}, {id: 1, distance: 4}},
	}
	assertCandidates(t, got, want)
}

func TestExactCandidatesEmptyAndSingleVector(t *testing.T) {
	got, err := exactCandidates(nil, 2, MetricL2, 3, 1)
	if err != nil {
		t.Fatalf("exactCandidates empty returned error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("len(empty candidates) = %d, want 0", len(got))
	}

	flat, dim, err := FlattenVectors([][]float32{{1, 2}}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	got, err = exactCandidates(flat, dim, MetricL2, 3, 1)
	if err != nil {
		t.Fatalf("exactCandidates single returned error: %v", err)
	}
	if len(got) != 1 || len(got[0]) != 0 {
		t.Fatalf("single-vector candidates = %v, want one empty list", got)
	}
}

func TestExactCandidatesRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name    string
		vectors []float32
		dim     int
		k       int
	}{
		{name: "bad dim", vectors: []float32{1, 2}, dim: 0, k: 1},
		{name: "unaligned storage", vectors: []float32{1, 2, 3}, dim: 2, k: 1},
		{name: "bad k", vectors: []float32{1, 2}, dim: 2, k: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := exactCandidates(tt.vectors, tt.dim, MetricL2, tt.k, 1); err == nil {
				t.Fatal("exactCandidates returned nil error")
			}
		})
	}
}

func TestExactCandidatesCosineUsesNormalizedVectors(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{10, 0},
		{1, 0},
		{0, 1},
	}, 0, MetricCosine)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := exactCandidates(flat, dim, MetricCosine, 2, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}
	want := [][]candidate{
		{{id: 1, distance: 0}, {id: 2, distance: 1}},
		{{id: 0, distance: 0}, {id: 2, distance: 1}},
		{{id: 0, distance: 1}, {id: 1, distance: 1}},
	}
	assertCandidates(t, got, want)
}

func TestApproximateKNNGCandidatesAreBoundedSortedAndDeterministic(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(64), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	left, err := approximateKNNGCandidates(flat, dim, MetricL2, 6, 42, 4, 1)
	if err != nil {
		t.Fatalf("approximateKNNGCandidates returned error: %v", err)
	}
	right, err := approximateKNNGCandidates(flat, dim, MetricL2, 6, 42, 4, 1)
	if err != nil {
		t.Fatalf("approximateKNNGCandidates second run returned error: %v", err)
	}
	assertCandidates(t, left, right)

	for sourceID, candidates := range left {
		if len(candidates) > 6 {
			t.Fatalf("len(candidates[%d]) = %d, want <= 6", sourceID, len(candidates))
		}
		for i, candidate := range candidates {
			if candidate.id == sourceID {
				t.Fatalf("candidates[%d] contains self: %v", sourceID, candidates)
			}
			if i > 0 && betterCandidate(candidate, candidates[i-1]) {
				t.Fatalf("candidates[%d] not sorted: %v", sourceID, candidates)
			}
		}
	}
}

func TestApproximateKNNGCandidatesRecallAgainstExact(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(80), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	approx, err := approximateKNNGCandidates(flat, dim, MetricL2, 8, 7, 6, 1)
	if err != nil {
		t.Fatalf("approximateKNNGCandidates returned error: %v", err)
	}
	exact, err := exactCandidates(flat, dim, MetricL2, 8, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}

	recall := candidateRecall(approx, exact, 8)
	if recall < 0.70 {
		t.Fatalf("candidate recall = %.3f, want >= 0.70", recall)
	}
}

func TestCandidateBuildersAreDeterministicAcrossWorkerCounts(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(96), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	exactSequential, err := exactCandidates(flat, dim, MetricL2, 8, 1)
	if err != nil {
		t.Fatalf("exactCandidates sequential returned error: %v", err)
	}
	exactParallel, err := exactCandidates(flat, dim, MetricL2, 8, 4)
	if err != nil {
		t.Fatalf("exactCandidates parallel returned error: %v", err)
	}
	assertCandidates(t, exactParallel, exactSequential)

	approxSequential, err := approximateKNNGCandidates(flat, dim, MetricL2, 8, 7, 4, 1)
	if err != nil {
		t.Fatalf("approximateKNNGCandidates sequential returned error: %v", err)
	}
	approxParallel, err := approximateKNNGCandidates(flat, dim, MetricL2, 8, 7, 4, 4)
	if err != nil {
		t.Fatalf("approximateKNNGCandidates parallel returned error: %v", err)
	}
	assertCandidates(t, approxParallel, approxSequential)
}

func TestInitialApproxNeighborIDsUseSeed(t *testing.T) {
	leftCollector := newCandidateIDCollector(128, 12)
	rightCollector := newCandidateIDCollector(128, 12)
	left := initialApproxNeighborIDs(128, 5, 12, 1, &leftCollector)
	right := initialApproxNeighborIDs(128, 5, 12, 2, &rightCollector)
	if sameIDSet(left, right) {
		t.Fatal("initialApproxNeighborIDs returned identical sets for different seeds")
	}
}

func TestCandidateIDCollectorDeduplicatesAndExcludesSource(t *testing.T) {
	collector := newCandidateIDCollector(8, 4)
	collector.reset()

	collector.add(1, 2)
	collector.add(1, 2)
	collector.add(2, 2)
	collector.addCandidates(2, []candidate{{id: 3}, {id: 3}, {id: 2}, {id: 4}})

	want := []int{1, 3, 4}
	if len(collector.ids) != len(want) {
		t.Fatalf("collector ids = %v, want %v", collector.ids, want)
	}
	for i, id := range want {
		if collector.ids[i] != id {
			t.Fatalf("collector ids = %v, want %v", collector.ids, want)
		}
	}
}

func assertCandidates(t *testing.T, got, want [][]candidate) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(candidates) = %d, want %d: %v", len(got), len(want), got)
	}
	for sourceID := range want {
		if len(got[sourceID]) != len(want[sourceID]) {
			t.Fatalf("len(candidates[%d]) = %d, want %d: %v", sourceID, len(got[sourceID]), len(want[sourceID]), got[sourceID])
		}
		for i := range want[sourceID] {
			if got[sourceID][i].id != want[sourceID][i].id {
				t.Fatalf("candidates[%d][%d].id = %d, want %d", sourceID, i, got[sourceID][i].id, want[sourceID][i].id)
			}
			if !almostEqual(got[sourceID][i].distance, want[sourceID][i].distance) {
				t.Fatalf("candidates[%d][%d].distance = %v, want %v", sourceID, i, got[sourceID][i].distance, want[sourceID][i].distance)
			}
		}
	}
}

func candidateRecall(got [][]candidate, want [][]candidate, k int) float64 {
	var hits int
	var total int
	for sourceID := range want {
		sourceHits, sourceTotal := eval.CountHitsAtK(candidateIDs(got[sourceID]), candidateIDs(want[sourceID]), k)
		hits += sourceHits
		total += sourceTotal
	}
	if total == 0 {
		return 1
	}
	return float64(hits) / float64(total)
}

func sameIDSet(left []int, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	seen := make(map[int]struct{}, len(left))
	for _, id := range left {
		seen[id] = struct{}{}
	}
	for _, id := range right {
		if _, ok := seen[id]; !ok {
			return false
		}
	}
	return true
}
