//go:build amd64 && !purego

package core

import "golang.org/x/sys/cpu"

const avx2DistanceMinDim = 16

// squaredL2 uses AVX2 on capable amd64 CPUs and falls back to portable Go
// otherwise. The size guard avoids paying assembly call overhead for very short
// vectors where scalar Go is already cheap.
func squaredL2(a, b []float32) float32 {
	if cpu.X86.HasAVX2 && len(a) >= avx2DistanceMinDim {
		return squaredL2AVX2(a, b)
	}
	return squaredL2Generic(a, b)
}

// dot uses AVX2 on capable amd64 CPUs and falls back to portable Go otherwise.
func dot(a, b []float32) float32 {
	if cpu.X86.HasAVX2 && len(a) >= avx2DistanceMinDim {
		return dotAVX2(a, b)
	}
	return dotGeneric(a, b)
}

func squaredL2AVX2(a, b []float32) float32
func dotAVX2(a, b []float32) float32
