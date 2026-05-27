package eval

import "testing"

func TestRecallAtK(t *testing.T) {
	got := []int{1, 2, 3}
	truth := []int{2, 4, 1}

	recall := RecallAtK(got, truth, 3)
	if recall != float64(2)/3 {
		t.Fatalf("RecallAtK = %v, want %v", recall, float64(2)/3)
	}
}

func TestRecallAtKHandlesShortInputs(t *testing.T) {
	got := []int{1}
	truth := []int{1, 2}

	recall := RecallAtK(got, truth, 3)
	if recall != float64(1)/3 {
		t.Fatalf("RecallAtK = %v, want %v", recall, float64(1)/3)
	}
}

func TestRecallAtKHandlesEmptyCutoff(t *testing.T) {
	if recall := RecallAtK(nil, nil, 0); recall != 1 {
		t.Fatalf("RecallAtK = %v, want 1", recall)
	}
}

func TestCountHitsAtKReturnsAvailableTruthTotal(t *testing.T) {
	hits, total := CountHitsAtK([]int{1, 4, 5}, []int{1, 2}, 4)
	if hits != 1 || total != 2 {
		t.Fatalf("CountHitsAtK = (%d, %d), want (1, 2)", hits, total)
	}
}
