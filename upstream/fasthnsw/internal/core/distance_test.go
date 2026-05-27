package core

import (
	"math"
	"testing"
)

func TestSquaredL2(t *testing.T) {
	got := squaredL2([]float32{1, 2, 3}, []float32{4, 2, -1})
	const want float32 = 25
	if got != want {
		t.Fatalf("squaredL2 = %v, want %v", got, want)
	}
}

func TestDistanceKernelsMatchGeneric(t *testing.T) {
	for _, dim := range []int{1, 2, 7, 8, 15, 16, 31, 32, 127, 128, 129} {
		left := make([]float32, dim)
		right := make([]float32, dim)
		for i := range left {
			left[i] = float32((i%17)-8) * 0.125
			right[i] = float32((i%13)-6) * 0.0625
		}

		if got, want := squaredL2(left, right), squaredL2Generic(left, right); !closeFloat32(got, want, 1e-4) {
			t.Fatalf("dim %d squaredL2 = %v, want generic %v", dim, got, want)
		}
		if got, want := dot(left, right), dotGeneric(left, right); !closeFloat32(got, want, 1e-4) {
			t.Fatalf("dim %d dot = %v, want generic %v", dim, got, want)
		}
	}
}

func TestNormalizeInto(t *testing.T) {
	dst := make([]float32, 2)
	if err := normalizeInto(dst, []float32{3, 4}); err != nil {
		t.Fatalf("normalizeInto returned error: %v", err)
	}

	if !almostEqual(dst[0], 0.6) || !almostEqual(dst[1], 0.8) {
		t.Fatalf("normalized vector = %v, want [0.6 0.8]", dst)
	}
}

func TestNormalizeIntoRejectsZeroVector(t *testing.T) {
	err := normalizeInto(make([]float32, 2), []float32{0, 0})
	if err == nil {
		t.Fatal("normalizeInto returned nil error")
	}
}

func TestCosineDistanceNormalized(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	if got := cosineDistanceNormalized(a, b); got != 1 {
		t.Fatalf("orthogonal cosine distance = %v, want 1", got)
	}

	if got := cosineDistanceNormalized(a, a); got != 0 {
		t.Fatalf("identical cosine distance = %v, want 0", got)
	}
}

func closeFloat32(got float32, want float32, tolerance float64) bool {
	return math.Abs(float64(got-want)) <= tolerance
}
