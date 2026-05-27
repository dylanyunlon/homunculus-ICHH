package core

import (
	"fmt"
	"math"
	"sort"
)

const maxAlphaDegrees = 180

// rngPrune applies the RNG pruning rule used by NSG/HNSW construction.
//
// This is an internal hot-path helper. It assumes vector storage, metric,
// sourceID, maxDegree, and prune mode were validated by the construction
// boundary. Candidate ids are still checked because candidate pools are mutable
// construction data.
//
// Candidates are processed from nearest to farthest. A candidate v is rejected
// when an already accepted neighbor w dominates it: dist(u,w) < dist(u,v) and
// dist(v,w) < dist(u,v). The first condition is usually implied by processing
// order, but it is checked explicitly to preserve the strict RNG definition
// under equal-distance ties.
func rngPrune(sourceID int, candidates []candidate, maxDegree int, vectors []float32, dim int, metric Metric) ([]candidate, error) {
	return pruneCandidates(sourceID, candidates, maxDegree, vectors, dim, metric, nil)
}

// alphaPrune applies the paper's alpha-pruning extension.
//
// This is an internal hot-path helper and assumes scalar/vector invariants were
// validated by the construction boundary. Alpha-pruning starts from RNG
// dominance and adds the geometric condition angle u-w-v > alpha, where the
// angle is measured at the already accepted neighbor w. The implementation
// compares cosines instead of calling acos: for alpha in [60, 180], angle >
// alpha iff cos(angle) < cos(alpha). alpha=60 is RNG-compatible for the
// standard RNG triangle case.
func alphaPrune(sourceID int, candidates []candidate, maxDegree int, alphaDegrees float64, vectors []float32, dim int, metric Metric) ([]candidate, error) {
	cosThreshold := math.Cos(alphaDegrees * math.Pi / 180)
	return pruneCandidates(sourceID, candidates, maxDegree, vectors, dim, metric, func(candidateID int, acceptedID int) bool {
		return angleUWVGreaterThanAlpha(sourceID, acceptedID, candidateID, vectors, dim, cosThreshold)
	})
}

func pruneCandidates(sourceID int, candidates []candidate, maxDegree int, vectors []float32, dim int, metric Metric, angleFilter func(candidateID int, acceptedID int) bool) ([]candidate, error) {
	normalized, err := normalizePruneCandidates(sourceID, candidates, vectors, dim, metric)
	if err != nil {
		return nil, err
	}

	pruned := make([]candidate, 0, maxDegree)
	for _, next := range normalized {
		dominated := false
		for _, accepted := range pruned {
			if accepted.distance >= next.distance {
				continue
			}
			if distance(metric, vectorAt(vectors, dim, next.id), vectorAt(vectors, dim, accepted.id)) >= next.distance {
				continue
			}
			if angleFilter != nil && !angleFilter(next.id, accepted.id) {
				continue
			}
			dominated = true
			break
		}
		if dominated {
			continue
		}

		pruned = append(pruned, next)
		if len(pruned) >= maxDegree {
			break
		}
	}
	return pruned, nil
}

// normalizePruneCandidates trusts vector storage and sourceID but still
// validates candidate ids because candidate pools are refined and merged
// repeatedly during construction.
func normalizePruneCandidates(sourceID int, candidates []candidate, vectors []float32, dim int, metric Metric) ([]candidate, error) {
	count := len(vectors) / dim
	source := vectorAt(vectors, dim, sourceID)
	seen := make(map[int]candidate, len(candidates))
	for _, next := range candidates {
		if next.id == sourceID {
			continue
		}
		if next.id < 0 || next.id >= count {
			return nil, fmt.Errorf("fasthnsw: candidate id %d out of range [0,%d)", next.id, count)
		}
		normalized := candidate{
			id:       next.id,
			distance: distance(metric, source, vectorAt(vectors, dim, next.id)),
		}
		if existing, ok := seen[next.id]; !ok || betterCandidate(normalized, existing) {
			seen[next.id] = normalized
		}
	}

	out := make([]candidate, 0, len(seen))
	for _, next := range seen {
		out = append(out, next)
	}
	sort.Slice(out, func(i, j int) bool {
		return betterCandidate(out[i], out[j])
	})
	return out, nil
}

// angleUWVGreaterThanAlpha reports whether angle u-w-v is greater than the
// alpha threshold represented by cosAlpha. Degenerate zero-length angle arms
// are treated as not satisfying the angle condition, which avoids pruning on
// undefined geometry.
func angleUWVGreaterThanAlpha(sourceID int, acceptedID int, candidateID int, vectors []float32, dim int, cosAlpha float64) bool {
	u := vectorAt(vectors, dim, sourceID)
	w := vectorAt(vectors, dim, acceptedID)
	v := vectorAt(vectors, dim, candidateID)

	var dotProduct float64
	var normUW float64
	var normVW float64
	for i := 0; i < dim; i++ {
		uw := float64(u[i] - w[i])
		vw := float64(v[i] - w[i])
		dotProduct += uw * vw
		normUW += uw * uw
		normVW += vw * vw
	}
	if normUW == 0 || normVW == 0 {
		return false
	}

	cosAngle := dotProduct / math.Sqrt(normUW*normVW)
	return cosAngle < cosAlpha
}
