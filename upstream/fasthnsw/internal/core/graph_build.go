package core

type pruneKind int

const (
	pruneKindRNG pruneKind = iota
	pruneKindAlpha
)

// pruneMode selects the pruning rule used while assembling construction
// graphs. RNG mode is used for final HNSW layers; alpha mode is used for
// intermediate graphs in the paper's refinement-before-search construction.
type pruneMode struct {
	kind         pruneKind
	alphaDegrees float64
}

// rngPruneMode returns the final-layer RNG pruning mode.
func rngPruneMode() pruneMode {
	return pruneMode{kind: pruneKindRNG}
}

// alphaPruneMode returns the intermediate alpha-pruning mode.
func alphaPruneMode(alphaDegrees float64) pruneMode {
	return pruneMode{kind: pruneKindAlpha, alphaDegrees: alphaDegrees}
}

// buildPrunedLayer converts per-node candidate lists into one adjacency layer.
//
// The assembly mirrors the graph-construction flow used by NSG/HNSW-style
// builders: prune each node's forward candidates, add retained edges in reverse
// direction, then prune again so reverse-edge repair does not violate the
// degree bound. The output shape matches Index.layers[layer]: adjacency[nodeID]
// is a deterministic list of neighbor ids ordered by candidate distance then id.
// Only node-local pruning is parallelized; reverse-edge merging stays serial so
// append order is independent of worker scheduling.
func buildPrunedLayer(candidates [][]candidate, maxDegree int, mode pruneMode, vectors []float32, dim int, metric Metric, workers int) ([][]int, error) {
	count := len(vectors) / dim
	adjacency := make([][]int, count)
	if count == 0 {
		return adjacency, nil
	}

	forward := make([][]candidate, count)
	err := parallelForNodes(count, workers, func(_ int, sourceID int) error {
		pruned, err := applyPruneMode(sourceID, candidates[sourceID], maxDegree, mode, vectors, dim, metric)
		if err != nil {
			return err
		}
		forward[sourceID] = pruned
		return nil
	})
	if err != nil {
		return nil, err
	}

	merged := make([][]candidate, count)
	for sourceID, pruned := range forward {
		merged[sourceID] = append(merged[sourceID], pruned...)
		for _, neighbor := range pruned {
			merged[neighbor.id] = append(merged[neighbor.id], candidate{id: sourceID})
		}
	}

	err = parallelForNodes(count, workers, func(_ int, sourceID int) error {
		pruned, err := applyPruneMode(sourceID, merged[sourceID], maxDegree, mode, vectors, dim, metric)
		if err != nil {
			return err
		}
		adjacency[sourceID] = candidateIDs(pruned)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return adjacency, nil
}

func applyPruneMode(sourceID int, candidates []candidate, maxDegree int, mode pruneMode, vectors []float32, dim int, metric Metric) ([]candidate, error) {
	switch mode.kind {
	case pruneKindRNG:
		return rngPrune(sourceID, candidates, maxDegree, vectors, dim, metric)
	case pruneKindAlpha:
		return alphaPrune(sourceID, candidates, maxDegree, mode.alphaDegrees, vectors, dim, metric)
	default:
		panic("fasthnsw: unsupported prune mode")
	}
}

func candidateIDs(candidates []candidate) []int {
	ids := make([]int, len(candidates))
	for i, candidate := range candidates {
		ids[i] = candidate.id
	}
	return ids
}
