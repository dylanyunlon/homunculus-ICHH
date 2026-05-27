package core

import (
	"fmt"
	"slices"
)

// BuildStandardHNSWForBenchmark builds a standard incremental HNSW index for
// internal algorithm comparisons. It is intentionally not exposed by the public
// facade package; public users should continue to call Index.Build, which uses
// the FastHNSW construction path.
func BuildStandardHNSWForBenchmark(cfg Config, vectors [][]float32) (*Index, error) {
	cfg = standardHNSWComparableConfig(cfg)
	idx, err := New(cfg)
	if err != nil {
		return nil, err
	}
	flat, dim, err := FlattenVectors(vectors, idx.cfg.Dim, idx.cfg.Metric)
	if err != nil {
		return nil, err
	}
	idx.vectors = flat
	idx.dim = dim
	idx.count = len(vectors)
	idx.cfg.Dim = dim
	idx.resetSearchableGraph()
	if err := idx.buildStandardHNSWGraph(); err != nil {
		idx.resetSearchableGraph()
		return nil, err
	}
	return idx, nil
}

func standardHNSWComparableConfig(cfg Config) Config {
	constructionL := cfg.ConstructionL
	if constructionL == 0 {
		constructionL = cfg.EfConstruction
	}
	if constructionL > 0 && cfg.CandidateK > constructionL {
		// CandidateK is FastHNSW-specific. Standard HNSW only needs
		// efConstruction, so internal baseline callers should not have to tune
		// unrelated FastHNSW candidate fields to run a fair comparison.
		cfg.CandidateK = constructionL
	}
	return cfg
}

// buildStandardHNSWGraph implements the original incremental HNSW insertion
// algorithm. Nodes are inserted in input-id order, upper layers use greedy
// ef=1 descent, and construction layers use efConstruction search followed by
// Algorithm 4's neighbor-selection heuristic.
func (idx *Index) buildStandardHNSWGraph() error {
	if idx.count == 0 {
		return fmt.Errorf("fasthnsw: vectors must not be empty")
	}

	levels := assignLevels(idx.count, idx.cfg.M, idx.cfg.Seed)
	maxAllocatedLayer := maxAssignedLayer(levels)
	layers := make([][][]int, maxAllocatedLayer+1)
	for layer := range layers {
		layers[layer] = make([][]int, idx.count)
	}

	efConstruction := idx.cfg.ConstructionL
	if efConstruction < idx.cfg.M {
		efConstruction = idx.cfg.M
	}

	entryPoint := 0
	maxLayer := levels[0]
	var scratch graphSearchScratch
	for nodeID := 1; nodeID < idx.count; nodeID++ {
		nodeLevel := levels[nodeID]
		query := vectorAt(idx.vectors, idx.dim, nodeID)
		currentEntries := []int{entryPoint}

		for layer := maxLayer; layer > nodeLevel; layer-- {
			results := graphSearchLayerFromEntries(layers[layer], idx.vectors, idx.dim, idx.cfg.Metric, currentEntries, query, 1, &scratch)
			if len(results) > 0 {
				currentEntries = resultIDs(results)
			}
		}

		minLayer := nodeLevel
		if minLayer > maxLayer {
			minLayer = maxLayer
		}
		for layer := minLayer; layer >= 0; layer-- {
			candidates := graphSearchLayerFromEntries(layers[layer], idx.vectors, idx.dim, idx.cfg.Metric, currentEntries, query, efConstruction, &scratch)
			selected, err := selectStandardHNSWNeighbors(nodeID, candidates, standardLayerMaxDegree(idx.cfg, layer), layers[layer], idx.vectors, idx.dim, idx.cfg.Metric, false, false)
			if err != nil {
				return err
			}
			if err := connectStandardHNSWNeighbors(layers[layer], nodeID, selected, standardLayerMaxDegree(idx.cfg, layer), idx.vectors, idx.dim, idx.cfg.Metric); err != nil {
				return err
			}
			if len(candidates) > 0 {
				currentEntries = resultIDs(candidates)
			}
		}

		if nodeLevel > maxLayer {
			entryPoint = nodeID
			maxLayer = nodeLevel
		}
	}

	idx.levels = levels
	idx.layers = layers
	idx.entryPoint = entryPoint
	idx.maxLayer = maxLayer
	idx.graphReady = true
	return nil
}

func standardLayerMaxDegree(cfg Config, layer int) int {
	if layer == 0 {
		return baseLayerMaxDegree(cfg)
	}
	return cfg.M
}

func connectStandardHNSWNeighbors(adjacency [][]int, sourceID int, neighbors []int, maxDegree int, vectors []float32, dim int, metric Metric) error {
	adjacency[sourceID] = append(adjacency[sourceID][:0], neighbors...)
	for _, neighborID := range neighbors {
		if neighborID < 0 || neighborID >= len(adjacency) {
			return fmt.Errorf("fasthnsw: neighbor id %d out of range [0,%d)", neighborID, len(adjacency))
		}
		adjacency[neighborID] = appendUniqueInt(adjacency[neighborID], sourceID)
		if len(adjacency[neighborID]) > maxDegree {
			candidates := resultsFromNeighborIDs(neighborID, adjacency[neighborID], vectors, dim, metric)
			shrunk, err := selectStandardHNSWNeighbors(neighborID, candidates, maxDegree, adjacency, vectors, dim, metric, false, false)
			if err != nil {
				return err
			}
			adjacency[neighborID] = shrunk
		}
	}
	return nil
}

// selectStandardHNSWNeighbors implements Algorithm 4
// SELECT-NEIGHBORS-HEURISTIC. The default baseline passes false for
// extendCandidates and keepPrunedConnections, matching the paper's core
// diversity heuristic without hnswlib-specific engineering extensions.
func selectStandardHNSWNeighbors(sourceID int, candidates []Result, maxDegree int, adjacency [][]int, vectors []float32, dim int, metric Metric, extendCandidates bool, keepPrunedConnections bool) ([]int, error) {
	if maxDegree <= 0 {
		return nil, fmt.Errorf("fasthnsw: maxDegree must be positive")
	}

	candidateList, err := normalizeStandardHNSWCandidates(sourceID, candidates, adjacency, vectors, dim, metric, extendCandidates)
	if err != nil {
		return nil, err
	}

	selected := make([]Result, 0, maxDegree)
	pruned := make([]Result, 0)
	for _, candidate := range candidateList {
		good := true
		for _, existing := range selected {
			if distance(metric, vectorAt(vectors, dim, candidate.ID), vectorAt(vectors, dim, existing.ID)) < candidate.Distance {
				good = false
				break
			}
		}
		if good {
			selected = append(selected, candidate)
			if len(selected) >= maxDegree {
				return resultIDs(selected), nil
			}
			continue
		}
		if keepPrunedConnections {
			pruned = append(pruned, candidate)
		}
	}

	if keepPrunedConnections {
		for _, candidate := range pruned {
			selected = append(selected, candidate)
			if len(selected) >= maxDegree {
				break
			}
		}
	}
	return resultIDs(selected), nil
}

func normalizeStandardHNSWCandidates(sourceID int, candidates []Result, adjacency [][]int, vectors []float32, dim int, metric Metric, extendCandidates bool) ([]Result, error) {
	count := len(vectors) / dim
	seen := make(map[int]Result, len(candidates))
	add := func(id int) error {
		if id == sourceID {
			return nil
		}
		if id < 0 || id >= count {
			return fmt.Errorf("fasthnsw: candidate id %d out of range [0,%d)", id, count)
		}
		result := resultForNode(vectors, dim, metric, id, vectorAt(vectors, dim, sourceID))
		if existing, ok := seen[id]; !ok || betterResult(result, existing) {
			seen[id] = result
		}
		return nil
	}

	for _, candidate := range candidates {
		if err := add(candidate.ID); err != nil {
			return nil, err
		}
		if extendCandidates {
			if candidate.ID < 0 || candidate.ID >= len(adjacency) {
				return nil, fmt.Errorf("fasthnsw: candidate id %d out of range [0,%d)", candidate.ID, len(adjacency))
			}
			for _, neighborID := range adjacency[candidate.ID] {
				if err := add(neighborID); err != nil {
					return nil, err
				}
			}
		}
	}

	out := make([]Result, 0, len(seen))
	for _, candidate := range seen {
		out = append(out, candidate)
	}
	slices.SortFunc(out, func(a, b Result) int {
		return compareDistanceID(a.Distance, a.ID, b.Distance, b.ID)
	})
	return out, nil
}

func resultsFromNeighborIDs(sourceID int, neighborIDs []int, vectors []float32, dim int, metric Metric) []Result {
	query := vectorAt(vectors, dim, sourceID)
	results := make([]Result, 0, len(neighborIDs))
	for _, neighborID := range neighborIDs {
		if neighborID == sourceID {
			continue
		}
		results = append(results, resultForNode(vectors, dim, metric, neighborID, query))
	}
	return results
}

func resultIDs(results []Result) []int {
	ids := make([]int, len(results))
	for i, result := range results {
		ids[i] = result.ID
	}
	return ids
}

func appendUniqueInt(values []int, value int) []int {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}
