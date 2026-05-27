package core

import "testing"

func TestFlattenVectorsL2(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{{1, 2}, {3, 4}}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	if dim != 2 {
		t.Fatalf("dim = %d, want 2", dim)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if flat[i] != want[i] {
			t.Fatalf("flat[%d] = %v, want %v", i, flat[i], want[i])
		}
	}
}

func TestFlattenVectorsCosineNormalizes(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{{3, 4}}, 0, MetricCosine)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	if dim != 2 {
		t.Fatalf("dim = %d, want 2", dim)
	}
	if !almostEqual(flat[0], 0.6) || !almostEqual(flat[1], 0.8) {
		t.Fatalf("flat = %v, want normalized [0.6 0.8]", flat)
	}
}

func TestFlattenVectorsRejectsCosineZeroVector(t *testing.T) {
	_, _, err := FlattenVectors([][]float32{{1, 0}, {0, 0}}, 0, MetricCosine)
	if err == nil {
		t.Fatal("FlattenVectors returned nil error")
	}
}

func TestFlattenVectorsRejectsDimensionMismatch(t *testing.T) {
	_, _, err := FlattenVectors([][]float32{{1, 2}, {3}}, 0, MetricL2)
	if err == nil {
		t.Fatal("FlattenVectors returned nil error")
	}
}
