package synth

// UniformVectors returns deterministic pseudo-uniform vectors.
func UniformVectors(count int, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for id := 0; id < count; id++ {
		vector := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v := (id+1)*(d+5)*37 + (id%11)*17 + d*13
			vector[d] = float32(v%997) / 997
		}
		vectors[id] = vector
	}
	return vectors
}

// UniformQueries returns deterministic pseudo-uniform query vectors.
func UniformQueries(count int, dim int) [][]float32 {
	queries := make([][]float32, count)
	for id := 0; id < count; id++ {
		query := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v := (id+3)*(d+7)*29 + id*19 + d*23
			query[d] = float32(v%991) / 991
		}
		queries[id] = query
	}
	return queries
}

const (
	clusteredBaseJitter  = 0.006
	clusteredQueryJitter = 0.003
)

// ClusteredVectors returns deterministic clustered vectors for recall tests.
//
// The per-dimension jitter is derived from a hash of (id, dimension), not from
// a short modulo cycle. This keeps generated datasets reproducible while
// avoiding large groups of exactly duplicated vectors, which would make
// ID-based Recall@k undercount equally good nearest neighbors.
func ClusteredVectors(count int, dim int, clusters int) [][]float32 {
	vectors := make([][]float32, count)
	for id := 0; id < count; id++ {
		cluster := id % clusters
		vector := make([]float32, dim)
		for d := 0; d < dim; d++ {
			vector[d] = clusteredCoordinate(id, d, dim, clusters, cluster, clusteredBaseJitter, 0x9e3779b97f4a7c15)
		}
		vectors[id] = vector
	}
	return vectors
}

// ClusteredQueries returns deterministic clustered query vectors.
func ClusteredQueries(count int, dim int, clusters int) [][]float32 {
	queries := make([][]float32, count)
	for id := 0; id < count; id++ {
		cluster := id % clusters
		query := make([]float32, dim)
		for d := 0; d < dim; d++ {
			query[d] = clusteredCoordinate(id, d, dim, clusters, cluster, clusteredQueryJitter, 0xd1b54a32d192ed03)
		}
		queries[id] = query
	}
	return queries
}

func clusteredCoordinate(id int, dimension int, dim int, clusters int, cluster int, jitterScale float32, salt uint64) float32 {
	center := float32((cluster+1)*(dimension+2)) / float32(clusters+dim)
	jitter := (hashUnitFloat32(uint64(id), uint64(dimension), salt) - 0.5) * 2 * jitterScale
	return center + jitter
}

func hashUnitFloat32(id uint64, dimension uint64, salt uint64) float32 {
	x := salt
	x ^= id + 0x9e3779b97f4a7c15 + (x << 6) + (x >> 2)
	x ^= dimension + 0xbf58476d1ce4e5b9 + (x << 6) + (x >> 2)
	x = splitmix64(x)
	return float32(x>>40) / float32(1<<24)
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// SameVectors reports whether two vector collections are exactly identical.
func SameVectors(left [][]float32, right [][]float32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if len(left[i]) != len(right[i]) {
			return false
		}
		for d := range left[i] {
			if left[i][d] != right[i][d] {
				return false
			}
		}
	}
	return true
}
