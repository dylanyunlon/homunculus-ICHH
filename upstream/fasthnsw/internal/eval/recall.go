package eval

// RecallAtK returns the fraction of truth top-k ids found in got top-k ids.
// It treats duplicate ids in truth as one relevant item and preserves the
// repository-wide convention that k <= 0 has perfect empty-set recall.
func RecallAtK(got []int, truth []int, k int) float64 {
	if k <= 0 {
		return 1
	}
	hits, _ := CountHitsAtK(got, truth, k)
	return float64(hits) / float64(k)
}

// CountHitsAtK counts how many got top-k ids appear in truth top-k ids. The
// returned total is the number of truth ids considered, which can be smaller
// than k for tiny candidate sets.
func CountHitsAtK(got []int, truth []int, k int) (hits int, total int) {
	if k <= 0 {
		return 0, 0
	}

	truthIDs := make(map[int]bool, k)
	for i := 0; i < k && i < len(truth); i++ {
		truthIDs[truth[i]] = true
		total++
	}

	limit := k
	if limit > len(got) {
		limit = len(got)
	}
	for i := 0; i < limit; i++ {
		if truthIDs[got[i]] {
			hits++
		}
	}
	return hits, total
}
