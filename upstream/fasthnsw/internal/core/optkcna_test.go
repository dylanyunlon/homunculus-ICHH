package core

import "testing"

func TestOptKCNARefreshesCandidatesThroughGraphSearch(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := [][]candidate{
		{{id: 1}},
		{{id: 2}},
		nil,
	}

	got, err := optKCNA(candidates, optKCNAConfig{
		CandidateK:   2,
		SearchEf:     3,
		MaxDegree:    2,
		AlphaDegrees: 120,
	}, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("optKCNA returned error: %v", err)
	}

	assertCandidateList(t, got[0], []candidate{
		{id: 1, distance: 1},
		{id: 2, distance: 4},
	})
}

func TestOptKCNAOutputsSortedCandidatesWithoutSelf(t *testing.T) {
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

	got, err := optKCNA(candidates, optKCNAConfig{
		CandidateK:   2,
		SearchEf:     3,
		MaxDegree:    2,
		AlphaDegrees: 90,
	}, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("optKCNA returned error: %v", err)
	}

	for sourceID, sourceCandidates := range got {
		if len(sourceCandidates) > 2 {
			t.Fatalf("len(candidates[%d]) = %d, want <= 2", sourceID, len(sourceCandidates))
		}
		for i, next := range sourceCandidates {
			if next.id == sourceID {
				t.Fatalf("candidates[%d] contains self: %v", sourceID, sourceCandidates)
			}
			if i > 0 && betterCandidate(next, sourceCandidates[i-1]) {
				t.Fatalf("candidates[%d] not sorted: %v", sourceID, sourceCandidates)
			}
		}
	}
}

func TestOptKCNADoesNotBridgeDisconnectedComponents(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{10},
		{11},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := [][]candidate{
		{{id: 1}},
		{{id: 0}},
		{{id: 3}},
		{{id: 2}},
	}

	got, err := optKCNA(candidates, optKCNAConfig{
		CandidateK:   3,
		SearchEf:     4,
		MaxDegree:    2,
		AlphaDegrees: 120,
	}, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("optKCNA returned error: %v", err)
	}

	assertCandidateList(t, got[0], []candidate{{id: 1, distance: 1}})
	assertCandidateList(t, got[2], []candidate{{id: 3, distance: 1}})
}

func TestOptKCNAPreservesOrImprovesCandidateRecall(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(48), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	initial, err := approximateKNNGCandidates(flat, dim, MetricL2, 4, 5, 2, 1)
	if err != nil {
		t.Fatalf("approximateKNNGCandidates returned error: %v", err)
	}
	exact, err := exactCandidates(flat, dim, MetricL2, 6, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}

	refreshed, err := optKCNA(initial, optKCNAConfig{
		CandidateK:   6,
		SearchEf:     8,
		MaxDegree:    6,
		AlphaDegrees: 90,
	}, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("optKCNA returned error: %v", err)
	}

	before := candidateRecall(initial, exact, 6)
	after := candidateRecall(refreshed, exact, 6)
	if after < before {
		t.Fatalf("candidate recall decreased from %.3f to %.3f", before, after)
	}
}

func TestOptKCNARejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name       string
		candidates [][]candidate
		cfg        optKCNAConfig
		vectors    []float32
		dim        int
		metric     Metric
	}{
		{name: "bad CandidateK", candidates: make([][]candidate, 2), cfg: optKCNAConfig{CandidateK: 0, SearchEf: 1, MaxDegree: 1, AlphaDegrees: 60}, vectors: []float32{0, 1}, dim: 1, metric: MetricL2},
		{name: "bad SearchEf", candidates: make([][]candidate, 2), cfg: optKCNAConfig{CandidateK: 2, SearchEf: 1, MaxDegree: 1, AlphaDegrees: 60}, vectors: []float32{0, 1}, dim: 1, metric: MetricL2},
		{name: "bad MaxDegree", candidates: make([][]candidate, 2), cfg: optKCNAConfig{CandidateK: 1, SearchEf: 1, MaxDegree: 0, AlphaDegrees: 60}, vectors: []float32{0, 1}, dim: 1, metric: MetricL2},
		{name: "bad alpha", candidates: make([][]candidate, 2), cfg: optKCNAConfig{CandidateK: 1, SearchEf: 1, MaxDegree: 1, AlphaDegrees: 59}, vectors: []float32{0, 1}, dim: 1, metric: MetricL2},
		{name: "bad workers", candidates: make([][]candidate, 2), cfg: optKCNAConfig{CandidateK: 1, SearchEf: 1, MaxDegree: 1, AlphaDegrees: 60, Workers: -1}, vectors: []float32{0, 1}, dim: 1, metric: MetricL2},
		{name: "bad metric", candidates: make([][]candidate, 2), cfg: optKCNAConfig{CandidateK: 1, SearchEf: 1, MaxDegree: 1, AlphaDegrees: 60}, vectors: []float32{0, 1}, dim: 1, metric: Metric(99)},
		{name: "bad dim", candidates: make([][]candidate, 2), cfg: optKCNAConfig{CandidateK: 1, SearchEf: 1, MaxDegree: 1, AlphaDegrees: 60}, vectors: []float32{0, 1}, dim: 0, metric: MetricL2},
		{name: "unaligned storage", candidates: make([][]candidate, 1), cfg: optKCNAConfig{CandidateK: 1, SearchEf: 1, MaxDegree: 1, AlphaDegrees: 60}, vectors: []float32{0, 1, 2}, dim: 2, metric: MetricL2},
		{name: "bad candidate count", candidates: nil, cfg: optKCNAConfig{CandidateK: 1, SearchEf: 1, MaxDegree: 1, AlphaDegrees: 60}, vectors: []float32{0, 1}, dim: 1, metric: MetricL2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := optKCNA(tt.candidates, tt.cfg, tt.vectors, tt.dim, tt.metric); err == nil {
				t.Fatal("optKCNA returned nil error")
			}
		})
	}
}

func TestOptKCNARejectsBadCandidateID(t *testing.T) {
	_, err := optKCNA([][]candidate{{{id: 2}}, nil}, optKCNAConfig{
		CandidateK:   1,
		SearchEf:     1,
		MaxDegree:    1,
		AlphaDegrees: 60,
	}, []float32{0, 1}, 1, MetricL2)
	if err == nil {
		t.Fatal("optKCNA returned nil error")
	}
}
