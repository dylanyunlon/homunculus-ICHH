package core

import (
	"fmt"
)

// candidate is one construction-time neighbor candidate for a source node.
// It is intentionally separate from Result because construction code will add
// pruning and graph-building metadata around candidates while public Search
// results should remain a stable user-facing type.
type candidate struct {
	id       int
	distance float32
}

// exactCandidates computes exact per-node k-nearest candidate lists.
//
// This O(n^2) builder is the correctness baseline for pruning, IterNSG, recall
// checks, and tiny-layer fallbacks. It excludes each source node itself,
// returns candidates sorted by distance then id, and operates on the same flat
// vector storage used by the index. Workers only partition independent source
// nodes, so output remains source-id ordered.
func exactCandidates(vectors []float32, dim int, metric Metric, k int, workers int) ([][]candidate, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("fasthnsw: vector dimension must be positive")
	}
	if len(vectors)%dim != 0 {
		return nil, fmt.Errorf("fasthnsw: flat vector storage is not aligned to dimension")
	}
	if k <= 0 {
		return nil, fmt.Errorf("fasthnsw: k must be positive")
	}

	count := len(vectors) / dim
	out := make([][]candidate, count)
	if count <= 1 {
		return out, nil
	}

	limit := k
	if limit > count-1 {
		limit = count - 1
	}

	err := parallelForNodes(count, workers, func(_ int, sourceID int) error {
		out[sourceID] = exactCandidatesForNode(vectors, dim, metric, sourceID, limit)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// exactCandidatesForNode scans all other vectors and keeps the nearest limit
// candidates for one source node.
func exactCandidatesForNode(vectors []float32, dim int, metric Metric, sourceID int, limit int) []candidate {
	source := vectorAt(vectors, dim, sourceID)
	best := make(candidateMaxHeap, 0, limit)

	count := len(vectors) / dim
	for id := 0; id < count; id++ {
		if id == sourceID {
			continue
		}
		next := candidate{
			id:       id,
			distance: distance(metric, source, vectorAt(vectors, dim, id)),
		}
		if best.Len() < limit {
			best.push(next)
			continue
		}
		if betterCandidate(next, best.worst()) {
			best.replaceWorst(next)
		}
	}

	return best.sorted()
}

// approximateKNNGCandidates builds the initial KNNG used by FastHNSW
// construction.
//
// The paper uses an approximate KNNG as the first construction phase rather
// than exact all-pairs neighbors. This pure Go implementation follows the same
// NN-Descent style idea without copying the official artifact: initialize each
// node with deterministic random neighbors, repeatedly exchange candidates
// through forward and reverse neighborhoods, then keep the nearest k candidates.
// Exact candidates are used only when k already covers the whole tiny layer.
// Workers only partition node-local work; candidate lists are still stored in
// source-id order for deterministic downstream pruning.
func approximateKNNGCandidates(vectors []float32, dim int, metric Metric, k int, seed int64, iterations int, workers int) ([][]candidate, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("fasthnsw: vector dimension must be positive")
	}
	if len(vectors)%dim != 0 {
		return nil, fmt.Errorf("fasthnsw: flat vector storage is not aligned to dimension")
	}
	if k <= 0 {
		return nil, fmt.Errorf("fasthnsw: k must be positive")
	}

	count := len(vectors) / dim
	out := make([][]candidate, count)
	if count <= 1 {
		return out, nil
	}

	limit := k
	if limit > count-1 {
		limit = count - 1
	}
	if count <= limit+1 {
		return exactCandidates(vectors, dim, metric, limit, workers)
	}

	workerCount := effectiveWorkerCount(workers, count)
	collectors := makeCandidateIDCollectors(workerCount, count, limit*(limit+2))
	err := parallelForNodes(count, workers, func(workerID int, sourceID int) error {
		initialIDs := initialApproxNeighborIDs(count, sourceID, limit, seed, &collectors[workerID])
		out[sourceID] = candidatesFromIDs(sourceID, initialIDs, limit, vectors, dim, metric)
		return nil
	})
	if err != nil {
		return nil, err
	}

	rounds := iterations
	if rounds < 1 {
		rounds = 1
	}
	for round := 0; round < rounds; round++ {
		reverse := reverseCandidateIDs(out, count)
		next := make([][]candidate, count)
		err := parallelForNodes(count, workers, func(workerID int, sourceID int) error {
			collector := &collectors[workerID]
			collector.reset()
			collector.addCandidates(sourceID, out[sourceID])
			for _, reverseID := range reverse[sourceID] {
				collector.add(reverseID, sourceID)
			}
			for _, current := range out[sourceID] {
				collector.addCandidates(sourceID, out[current.id])
			}
			for _, reverseID := range reverse[sourceID] {
				collector.addCandidates(sourceID, out[reverseID])
			}
			next[sourceID] = candidatesFromIDs(sourceID, collector.ids, limit, vectors, dim, metric)
			return nil
		})
		if err != nil {
			return nil, err
		}
		out = next
	}
	return out, nil
}

// initialApproxNeighborIDs writes deterministic seed-dependent initializer ids
// into collector and returns its current id slice. The initializer combines
// nearby ids, which help ordered inputs, with pseudo-random ids, which keep the
// approximate KNNG from depending on input order alone. The collector is reused
// across source nodes to avoid map allocation in this construction hot path.
func initialApproxNeighborIDs(count int, sourceID int, limit int, seed int64, collector *candidateIDCollector) []int {
	collector.reset()

	// Adjacent ids are cheap deterministic seeds and are useful when input ids
	// already carry locality, while random ids keep the initializer from
	// degenerating on shuffled datasets.
	localBudget := limit / 2
	for offset := 1; len(collector.ids) < localBudget && offset < count; offset++ {
		collector.add((sourceID+offset)%count, sourceID)
		if len(collector.ids) >= localBudget {
			break
		}
		collector.add((sourceID-offset+count)%count, sourceID)
	}

	stream := uint64(mixSeed(seed, int64(sourceID), int64(count)))
	maxAttempts := uint64(count * 4)
	for attempt := uint64(0); len(collector.ids) < limit && attempt < maxAttempts; attempt++ {
		id := int(splitmix64(stream+attempt*0x9e3779b97f4a7c15) % uint64(count))
		collector.add(id, sourceID)
	}

	if len(collector.ids) < limit {
		start := int(splitmix64(stream^0xd1b54a32d192ed03) % uint64(count))
		for offset := 0; len(collector.ids) < limit && offset < count; offset++ {
			collector.add((start+offset)%count, sourceID)
		}
	}
	return collector.ids
}

// reverseCandidateIDs builds R(v), the reverse-neighbor lists used by the
// NN-Descent style exchange step. It uses one compact backing array instead of
// per-node append growth so construction cost scales with the edge count rather
// than with many small slice allocations.
func reverseCandidateIDs(candidates [][]candidate, count int) [][]int {
	offsets := make([]int, count+1)
	for _, sourceCandidates := range candidates {
		for _, candidate := range sourceCandidates {
			offsets[candidate.id+1]++
		}
	}
	for id := 1; id <= count; id++ {
		offsets[id] += offsets[id-1]
	}

	storage := make([]int, offsets[count])
	positions := make([]int, count)
	copy(positions, offsets[:count])
	reverse := make([][]int, count)

	for sourceID, sourceCandidates := range candidates {
		for _, candidate := range sourceCandidates {
			position := positions[candidate.id]
			storage[position] = sourceID
			positions[candidate.id]++
		}
	}
	for id := 0; id < count; id++ {
		reverse[id] = storage[offsets[id]:offsets[id+1]]
	}
	return reverse
}

// candidateIDCollector is a reusable set for construction-time neighbor ids.
// The mark table avoids per-node maps while preserving deterministic candidate
// expansion: ids are emitted in discovery order and later sorted by distance.
type candidateIDCollector struct {
	marks []uint32
	mark  uint32
	ids   []int
}

func newCandidateIDCollector(count int, capacity int) candidateIDCollector {
	collector := candidateIDCollector{}
	collector.reserve(count, capacity)
	return collector
}

// makeCandidateIDCollectors creates one reusable collector per worker. Sharing
// collectors across goroutines would corrupt mark-table generations, so
// construction parallelism keeps this scratch worker-local.
func makeCandidateIDCollectors(workers int, count int, capacity int) []candidateIDCollector {
	collectors := make([]candidateIDCollector, workers)
	for i := range collectors {
		collectors[i].reserve(count, capacity)
	}
	return collectors
}

// reserve prepares the collector for a graph with count local nodes and a
// typical per-source capacity. Existing scratch is reused when it already fits.
func (c *candidateIDCollector) reserve(count int, capacity int) {
	if len(c.marks) != count {
		c.marks = make([]uint32, count)
		c.mark = 0
	}
	if cap(c.ids) < capacity {
		c.ids = make([]int, 0, capacity)
	} else {
		c.ids = c.ids[:0]
	}
}

// reset starts a new logical set without clearing the mark table on every
// source node. Full clearing is needed only if the generation counter wraps.
func (c *candidateIDCollector) reset() {
	c.mark++
	if c.mark == 0 {
		clear(c.marks)
		c.mark = 1
	}
	c.ids = c.ids[:0]
}

// add inserts one valid local id unless it is the source node or was already
// seen in the current collector generation.
func (c *candidateIDCollector) add(id int, sourceID int) {
	if id == sourceID || c.marks[id] == c.mark {
		return
	}
	c.marks[id] = c.mark
	c.ids = append(c.ids, id)
}

// addCandidates inserts candidate ids into the current collector generation.
func (c *candidateIDCollector) addCandidates(sourceID int, candidates []candidate) {
	for _, candidate := range candidates {
		c.add(candidate.id, sourceID)
	}
}

// candidatesFromIDs computes distances for collected ids and keeps the nearest
// limit candidates using a bounded max-heap.
func candidatesFromIDs(sourceID int, ids []int, limit int, vectors []float32, dim int, metric Metric) []candidate {
	source := vectorAt(vectors, dim, sourceID)
	best := make(candidateMaxHeap, 0, limit)
	for _, id := range ids {
		next := candidate{
			id:       id,
			distance: distance(metric, source, vectorAt(vectors, dim, id)),
		}
		if best.Len() < limit {
			best.push(next)
			continue
		}
		if betterCandidate(next, best.worst()) {
			best.replaceWorst(next)
		}
	}
	return best.sorted()
}

// mixSeed derives deterministic independent random streams from one user seed.
func mixSeed(seed int64, values ...int64) int64 {
	x := uint64(seed) + 0x9e3779b97f4a7c15
	for _, value := range values {
		x ^= uint64(value) + 0x9e3779b97f4a7c15 + (x << 6) + (x >> 2)
	}
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return int64(x)
}

// splitmix64 is a small deterministic mixer used to derive pseudo-random ids
// without allocating a per-node math/rand source during KNNG initialization.
func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// betterCandidate reports whether a should sort before b in candidate order.
// Candidate order is deterministic: distance ascending, then smaller id.
func betterCandidate(a, b candidate) bool {
	return compareDistanceID(a.distance, a.id, b.distance, b.id) < 0
}

// worseCandidate reports whether a should sort after b in candidate order.
func worseCandidate(a, b candidate) bool {
	return compareDistanceID(a.distance, a.id, b.distance, b.id) > 0
}
