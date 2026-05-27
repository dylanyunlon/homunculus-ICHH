package core

import "testing"

func TestResultMinHeapPopsNearestFirst(t *testing.T) {
	candidates := resultMinHeap{
		{ID: 2, Distance: 0.5},
		{ID: 1, Distance: 0.5},
		{ID: 3, Distance: 0.1},
	}
	candidates.init()

	want := []Result{
		{ID: 3, Distance: 0.1},
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 0.5},
	}
	for i, expected := range want {
		got := candidates.pop()
		if got != expected {
			t.Fatalf("pop %d = %+v, want %+v", i, got, expected)
		}
	}
}

func TestResultMaxHeapPopsWorstFirst(t *testing.T) {
	results := resultMaxHeap{
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 0.5},
		{ID: 3, Distance: 0.1},
	}
	results.init()

	got := results.pop()
	want := Result{ID: 2, Distance: 0.5}
	if got != want {
		t.Fatalf("first pop = %+v, want worst result %+v", got, want)
	}
}

func TestResultMaxHeapReplaceWorstMaintainsOrdering(t *testing.T) {
	results := resultMaxHeap{
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 0.5},
		{ID: 3, Distance: 0.1},
	}
	results.init()

	results.replaceWorst(Result{ID: 4, Distance: 0.2})

	got := results.sorted()
	want := []Result{
		{ID: 3, Distance: 0.1},
		{ID: 4, Distance: 0.2},
		{ID: 1, Distance: 0.5},
	}
	assertResults(t, got, want)
}

func TestResultHeapsPushMaintainOrdering(t *testing.T) {
	var candidates resultMinHeap
	var results resultMaxHeap
	for _, result := range []Result{
		{ID: 2, Distance: 0.5},
		{ID: 1, Distance: 0.5},
		{ID: 3, Distance: 0.1},
	} {
		candidates.push(result)
		results.push(result)
	}

	if got, want := candidates.pop(), (Result{ID: 3, Distance: 0.1}); got != want {
		t.Fatalf("min heap pop = %+v, want %+v", got, want)
	}
	if got, want := results.pop(), (Result{ID: 2, Distance: 0.5}); got != want {
		t.Fatalf("max heap pop = %+v, want %+v", got, want)
	}
}

func TestResultMaxHeapSortedReturnsPublicOrder(t *testing.T) {
	results := resultMaxHeap{
		{ID: 2, Distance: 0.5},
		{ID: 1, Distance: 0.5},
		{ID: 3, Distance: 0.1},
	}

	got := results.sorted()
	want := []Result{
		{ID: 3, Distance: 0.1},
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 0.5},
	}
	assertResults(t, got, want)
}

func TestCandidateMaxHeapPopsWorstFirst(t *testing.T) {
	candidates := candidateMaxHeap{
		{id: 1, distance: 0.5},
		{id: 2, distance: 0.5},
		{id: 3, distance: 0.1},
	}
	candidates.init()

	got := candidates.pop()
	want := candidate{id: 2, distance: 0.5}
	if got != want {
		t.Fatalf("first pop = %+v, want worst candidate %+v", got, want)
	}
}

func TestCandidateMaxHeapReplaceWorstMaintainsOrdering(t *testing.T) {
	candidates := candidateMaxHeap{
		{id: 1, distance: 0.5},
		{id: 2, distance: 0.5},
		{id: 3, distance: 0.1},
	}
	candidates.init()

	candidates.replaceWorst(candidate{id: 4, distance: 0.2})

	got := candidates.sorted()
	want := []candidate{
		{id: 3, distance: 0.1},
		{id: 4, distance: 0.2},
		{id: 1, distance: 0.5},
	}
	assertHeapCandidateList(t, got, want)
}

func TestCandidateMaxHeapSortedReturnsConstructionOrder(t *testing.T) {
	candidates := candidateMaxHeap{
		{id: 2, distance: 0.5},
		{id: 1, distance: 0.5},
		{id: 3, distance: 0.1},
	}

	got := candidates.sorted()
	want := []candidate{
		{id: 3, distance: 0.1},
		{id: 1, distance: 0.5},
		{id: 2, distance: 0.5},
	}
	assertHeapCandidateList(t, got, want)
}

func assertHeapCandidateList(t *testing.T, got []candidate, want []candidate) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(candidates) = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("candidates[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}
