package core

import "testing"

func TestRNGPruneKeepsCandidatesWhenNoDominance(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{0, 1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := rngPrune(0, []candidate{{id: 1}, {id: 2}}, 4, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	want := []candidate{{id: 1, distance: 1}, {id: 2, distance: 1}}
	assertCandidateList(t, got, want)
}

func TestRNGPruneRemovesDominatedCollinearCandidate(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{2, 0},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := rngPrune(0, []candidate{{id: 2}, {id: 1}}, 4, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	want := []candidate{{id: 1, distance: 1}}
	assertCandidateList(t, got, want)
}

func TestRNGPruneHonorsMaxDegree(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{-1},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := rngPrune(0, []candidate{{id: 3}, {id: 2}, {id: 1}}, 1, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	want := []candidate{{id: 1, distance: 1}}
	assertCandidateList(t, got, want)
}

func TestPruneNormalizesSelfDuplicatesAndStaleDistances(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := rngPrune(0, []candidate{
		{id: 0, distance: -1},
		{id: 2, distance: 0},
		{id: 1, distance: 99},
		{id: 1, distance: 100},
	}, 4, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	want := []candidate{{id: 1, distance: 1}}
	assertCandidateList(t, got, want)
}

func TestAlphaPruneCanBeLessAggressiveThanRNG(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{1, 1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := []candidate{{id: 1}, {id: 2}}

	rng, err := rngPrune(0, candidates, 4, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	alpha60, err := alphaPrune(0, candidates, 4, 60, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("alphaPrune(60) returned error: %v", err)
	}
	alpha120, err := alphaPrune(0, candidates, 4, 120, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("alphaPrune(120) returned error: %v", err)
	}

	assertCandidateList(t, rng, []candidate{{id: 1, distance: 1}})
	assertCandidateList(t, alpha60, rng)
	assertCandidateList(t, alpha120, []candidate{{id: 1, distance: 1}, {id: 2, distance: 2}})
}

func TestPruneRejectsOutOfRangeCandidate(t *testing.T) {
	_, err := rngPrune(0, []candidate{{id: 2}}, 1, []float32{0, 1}, 1, MetricL2)
	if err == nil {
		t.Fatal("rngPrune returned nil error")
	}
}

func TestAngleUWVGreaterThanAlphaHandlesDegenerateGeometry(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{0, 0},
		{1, 0},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	if angleUWVGreaterThanAlpha(0, 1, 2, flat, dim, -0.5) {
		t.Fatal("degenerate angle unexpectedly satisfied alpha condition")
	}
}

func assertCandidateList(t *testing.T, got, want []candidate) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(candidates) = %d, want %d: got %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].id != want[i].id {
			t.Fatalf("candidate %d id = %d, want %d", i, got[i].id, want[i].id)
		}
		if !almostEqual(got[i].distance, want[i].distance) {
			t.Fatalf("candidate %d distance = %v, want %v", i, got[i].distance, want[i].distance)
		}
	}
}
