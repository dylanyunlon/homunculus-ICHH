package core

// Result is one nearest-neighbor result returned by Search.
type Result struct {
	ID       int
	Distance float32
}

// ResultIDs returns up to limit ids from already ordered results.
// It is used by validation and tests that compare result sets without
// duplicating result-to-id adapter loops.
func ResultIDs(results []Result, limit int) []int {
	if limit < 0 || limit > len(results) {
		limit = len(results)
	}
	ids := make([]int, limit)
	for i := 0; i < limit; i++ {
		ids[i] = results[i].ID
	}
	return ids
}

// betterResult reports whether a should sort before b in public result order.
// Distances sort ascending, and equal distances break ties by smaller id.
func betterResult(a, b Result) bool {
	return compareDistanceID(a.Distance, a.ID, b.Distance, b.ID) < 0
}

// worseResult reports whether a should sort after b in public result order.
// It is used by max-heaps that keep the farthest retained result at the root.
func worseResult(a, b Result) bool {
	return compareDistanceID(a.Distance, a.ID, b.Distance, b.ID) > 0
}

// compareDistanceID returns the deterministic nearest-neighbor ordering for
// two (distance, id) pairs: distance ascending, then id ascending.
func compareDistanceID(aDistance float32, aID int, bDistance float32, bID int) int {
	if aDistance < bDistance {
		return -1
	}
	if aDistance > bDistance {
		return 1
	}
	if aID < bID {
		return -1
	}
	if aID > bID {
		return 1
	}
	return 0
}
