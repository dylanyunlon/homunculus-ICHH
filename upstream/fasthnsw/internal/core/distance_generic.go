//go:build !amd64 || purego

package core

// squaredL2 uses the portable Go distance kernel on non-amd64 platforms and
// when users build with -tags purego.
func squaredL2(a, b []float32) float32 {
	return squaredL2Generic(a, b)
}

// dot uses the portable Go dot-product kernel on non-amd64 platforms and when
// users build with -tags purego.
func dot(a, b []float32) float32 {
	return dotGeneric(a, b)
}
