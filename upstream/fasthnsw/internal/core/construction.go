package core

import (
	"math"
	"math/rand"
	"sort"

	"github.com/cryo-zd/fasthnsw/internal/eval"
)

const minApproxKNNGIterations = 4

type candidateQualityConfig struct {
	TargetRecall float64
	Controls     int
	ControlIDs   []int
	Seed         int64
	K            int
	Workers      int
}

type candidateRefinementStats struct {
	InitialRecall float64
	FinalRecall   float64
	Iterations    int
}

type constructionLayout struct {
	labels      []int
	layerCounts []int
}

// buildSearchableGraph constructs all HNSW layers for the vectors already
// owned by idx. It follows the FastHNSW construction shape: assign every
// node's maximum layer first, globally relabel construction data by descending
// layer, then build layer i over the complete prefix D_i = {u | level(u) >= i}.
func (idx *Index) buildSearchableGraph() error {
	levels := assignLevels(idx.count, idx.cfg.M, idx.cfg.Seed)
	maxLayer := maxAssignedLayer(levels)
	layout := buildConstructionLayout(levels, maxLayer)
	constructionVectors := copyLayoutVectors(layout.labels, idx.vectors, idx.dim)
	entryPoint := selectEntryPointFromLayout(layout, maxLayer, idx.cfg.Seed)

	layers := make([][][]int, maxLayer+1)
	for layer := maxLayer; layer >= 0; layer-- {
		labels := layout.labels[:layout.layerCounts[layer]]
		layerVectors := constructionVectors[:len(labels)*idx.dim]
		maxDegree := idx.cfg.M
		if layer == 0 {
			maxDegree = baseLayerMaxDegree(idx.cfg)
		}
		adjacency, err := buildHNSWLayer(layer, labels, idx.count, layerVectors, idx.dim, idx.cfg, maxDegree)
		if err != nil {
			return err
		}
		layers[layer] = adjacency
	}

	idx.levels = levels
	idx.layers = layers
	idx.entryPoint = entryPoint
	idx.maxLayer = maxLayer
	idx.graphReady = true
	return nil
}

// buildConstructionLayout builds the clean-room equivalent of Algorithm 7's
// relabeling layout. labels maps construction-local ids to original node ids,
// and layerCounts[i] is the prefix length for D_i = {u | level(u) >= i}. The
// official artifact's graph_utils.h getMapping uses the same descending-level
// label order and cumulative per-layer counts; the public Go index still
// exposes and persists original ids.
func buildConstructionLayout(levels []int, maxLayer int) constructionLayout {
	labels := make([]int, 0, len(levels))
	layerCounts := make([]int, maxLayer+1)
	for layer := maxLayer; layer >= 0; layer-- {
		for id, assignedLayer := range levels {
			if assignedLayer == layer {
				labels = append(labels, id)
			}
		}
		layerCounts[layer] = len(labels)
	}

	return constructionLayout{
		labels:      labels,
		layerCounts: layerCounts,
	}
}

// copyLayoutVectors creates the construction-local vector buffer described by
// constructionLayout.labels. Keeping this buffer separate lets every FastHNSW
// layer use a contiguous prefix, as in Algorithm 7, without changing the
// library's external node ids or the stored vector order.
func copyLayoutVectors(labels []int, vectors []float32, dim int) []float32 {
	out := make([]float32, len(labels)*dim)
	for localID, globalID := range labels {
		copy(out[localID*dim:(localID+1)*dim], vectorAt(vectors, dim, globalID))
	}
	return out
}

// assignLevels samples each node's maximum HNSW layer using the standard
// exponential distribution. The configured seed is the only source of
// randomness, which keeps builds reproducible.
func assignLevels(count int, maxDegree int, seed int64) []int {
	levels := make([]int, count)
	if count == 0 || maxDegree <= 1 {
		// M=1 is a valid but sparse configuration. The logarithmic sampler is
		// undefined there, so construction falls back to a single-layer graph.
		return levels
	}

	rng := rand.New(rand.NewSource(seed))
	normalizer := math.Log(float64(maxDegree))
	for id := 0; id < count; id++ {
		u := rng.Float64()
		if u == 0 {
			u = math.SmallestNonzeroFloat64
		}
		levels[id] = int(-math.Log(u) / normalizer)
	}
	return levels
}

// maxAssignedLayer returns the highest sampled layer in a level table.
func maxAssignedLayer(levels []int) int {
	maxLayer := 0
	for _, level := range levels {
		if level > maxLayer {
			maxLayer = level
		}
	}
	return maxLayer
}

// selectEntryPointFromLayout picks a deterministic seeded-random node from the
// exact top-layer segment in a construction layout.
func selectEntryPointFromLayout(layout constructionLayout, layer int, seed int64) int {
	count := layout.layerCounts[layer]
	higherCount := 0
	if layer+1 < len(layout.layerCounts) {
		higherCount = layout.layerCounts[layer+1]
	}
	top := layout.labels[higherCount:count]
	if len(top) == 0 {
		return -1
	}
	rng := rand.New(rand.NewSource(mixSeed(seed, int64(layer), int64(len(top)))))
	return top[rng.Intn(len(top))]
}

// buildHNSWLayer returns one global adjacency table for a layer label set.
// Small layers are made complete because their degree is already within the
// layer's HNSW bound: M for upper layers and 2*M for layer 0. Larger layers use
// IterNSG and final RNG pruning. labels are original node ids ordered by
// construction-local id; vectors is the matching compact vector prefix.
func buildHNSWLayer(layer int, labels []int, totalCount int, vectors []float32, dim int, cfg Config, maxDegree int) ([][]int, error) {
	if len(labels)-1 <= maxDegree {
		return buildCompleteLayer(labels, totalCount, vectors, dim, cfg.Metric), nil
	}
	return buildIterNSGLayer(layer, labels, totalCount, vectors, dim, cfg, maxDegree)
}

// buildCompleteLayer connects every node in a tiny layer to every other layer
// node. Neighbor order still follows distance then id so search behavior stays
// deterministic even before pruning is needed.
func buildCompleteLayer(labels []int, totalCount int, vectors []float32, dim int, metric Metric) [][]int {
	adjacency := make([][]int, totalCount)
	for sourceLocalID, sourceID := range labels {
		source := vectorAt(vectors, dim, sourceLocalID)
		neighbors := make([]candidate, 0, len(labels)-1)
		for neighborLocalID, neighborID := range labels {
			if neighborID == sourceID {
				continue
			}
			neighbors = append(neighbors, candidate{
				id:       neighborID,
				distance: distance(metric, source, vectorAt(vectors, dim, neighborLocalID)),
			})
		}
		sort.Slice(neighbors, func(i, j int) bool {
			return betterCandidate(neighbors[i], neighbors[j])
		})
		adjacency[sourceID] = candidateIDs(neighbors)
	}
	return adjacency
}

// buildIterNSGLayer builds one non-trivial HNSW layer. Candidate ids are local
// while IterNSG runs, then the final RNG-pruned adjacency is mapped back to the
// index's global node ids. labels are the Algorithm 7 construction labels for
// this layer prefix, and vectors is the compact vector prefix in that order.
func buildIterNSGLayer(layer int, labels []int, totalCount int, vectors []float32, dim int, cfg Config, maxDegree int) ([][]int, error) {
	initialK := cfg.K0
	if initialK > len(labels)-1 {
		initialK = len(labels) - 1
	}
	candidateK := cfg.CandidateK
	if candidateK > len(labels)-1 {
		candidateK = len(labels) - 1
	}

	initialKNNG, err := approximateKNNGCandidates(
		vectors,
		dim,
		cfg.Metric,
		initialK,
		mixSeed(cfg.Seed, int64(layer), int64(len(labels))),
		approxKNNGIterations(cfg),
		cfg.Workers,
	)
	if err != nil {
		return nil, err
	}

	searchEf := cfg.ConstructionL
	if searchEf > len(labels)-1 {
		searchEf = len(labels) - 1
	}
	initialGraph := candidatesToAdjacency(initialKNNG)
	candidates := acquireCandidatesByGraphSearch(initialGraph, vectors, dim, cfg.Metric, candidateK, searchEf, cfg.Workers)

	refreshCfg := fastHNSWOptKCNAConfig(cfg, candidateK, searchEf, maxDegree)
	controlSeed := mixSeed(cfg.Seed, int64(layer), int64(candidateK))
	qualityCfg := candidateQualityConfig{
		TargetRecall: cfg.CandidateRecall,
		Controls:     cfg.CandidateControls,
		ControlIDs:   selectCandidateControls(len(labels), cfg.CandidateControls, controlSeed),
		Seed:         controlSeed,
		K:            candidateK,
		Workers:      cfg.Workers,
	}
	candidates, _, err = refineCandidatesUntilRecall(candidates, refreshCfg, qualityCfg, vectors, dim, cfg.Metric, cfg.Iterations)
	if err != nil {
		return nil, err
	}

	localAdjacency, err := buildFinalHNSWLayer(candidates, maxDegree, vectors, dim, cfg.Metric, cfg.Workers)
	if err != nil {
		return nil, err
	}
	return mapLayerAdjacencyToGlobal(localAdjacency, labels, totalCount), nil
}

func buildFinalHNSWLayer(candidates [][]candidate, maxDegree int, vectors []float32, dim int, metric Metric, workers int) ([][]int, error) {
	return buildPrunedLayer(candidates, maxDegree, rngPruneMode(), vectors, dim, metric, workers)
}

// fastHNSWOptKCNAConfig maps the public construction config to one FastHNSW
// candidate-refresh round.
//
// Section 5.3 states that each HNSW layer is an RNG index because it prunes
// k-CNA results without DFS connectivity enhancement. In the official
// artifact, KGraphImpl::build calls buildPG for each construction round, and
// buildPG skips tree_grow when pg_type is HNSW or NSW. This implementation
// follows that behavior without copying artifact source code.
func fastHNSWOptKCNAConfig(cfg Config, candidateK int, searchEf int, maxDegree int) optKCNAConfig {
	return optKCNAConfig{
		CandidateK:   candidateK,
		SearchEf:     searchEf,
		MaxDegree:    maxDegree,
		AlphaDegrees: cfg.Alpha,
		Workers:      cfg.Workers,
	}
}

func baseLayerMaxDegree(cfg Config) int {
	return 2 * cfg.M
}

// approxKNNGIterations keeps the KNNG initializer independent from IterNSG's
// refinement loop count while still deriving a deterministic effort level from
// the public config.
func approxKNNGIterations(cfg Config) int {
	rounds := cfg.Iterations * 2
	if rounds < minApproxKNNGIterations {
		return minApproxKNNGIterations
	}
	return rounds
}

func candidatesToAdjacency(candidates [][]candidate) [][]int {
	adjacency := make([][]int, len(candidates))
	for sourceID, sourceCandidates := range candidates {
		adjacency[sourceID] = candidateIDs(sourceCandidates)
	}
	return adjacency
}

// acquireCandidatesByGraphSearch implements the Algorithm 6 transition from an
// initial KNNG to k-CNA candidate sets: each node searches the current graph
// using its own vector as the query, then keeps the nearest k non-self results.
func acquireCandidatesByGraphSearch(adjacency [][]int, vectors []float32, dim int, metric Metric, candidateK int, searchEf int, workers int) [][]candidate {
	out := make([][]candidate, len(adjacency))
	workerCount := effectiveWorkerCount(workers, len(adjacency))
	scratches := makeGraphSearchScratches(workerCount, len(adjacency), searchEf)
	_ = parallelForNodes(len(adjacency), workers, func(workerID int, sourceID int) error {
		results := graphSearchLayer(adjacency, vectors, dim, metric, sourceID, vectorAt(vectors, dim, sourceID), searchEf, &scratches[workerID])
		out[sourceID] = candidatesFromResults(sourceID, results, candidateK)
		return nil
	})
	return out
}

// refineCandidatesUntilRecall follows Algorithm 6's stopping rule. Candidate
// quality is estimated before each OptKCNA round and after each refresh; the
// public Iterations config is only a maximum round cap, matching the official
// artifact's recall-threshold plus loop-cap structure.
func refineCandidatesUntilRecall(candidates [][]candidate, refreshCfg optKCNAConfig, qualityCfg candidateQualityConfig, vectors []float32, dim int, metric Metric, maxIterations int) ([][]candidate, candidateRefinementStats, error) {
	recall, err := estimateCandidateRecall(candidates, qualityCfg, vectors, dim, metric)
	if err != nil {
		return nil, candidateRefinementStats{}, err
	}
	stats := candidateRefinementStats{InitialRecall: recall, FinalRecall: recall}
	for stats.Iterations < maxIterations && stats.FinalRecall < qualityCfg.TargetRecall {
		candidates, err = optKCNA(candidates, refreshCfg, vectors, dim, metric)
		if err != nil {
			return nil, candidateRefinementStats{}, err
		}
		stats.Iterations++
		stats.FinalRecall, err = estimateCandidateRecall(candidates, qualityCfg, vectors, dim, metric)
		if err != nil {
			return nil, candidateRefinementStats{}, err
		}
	}
	return candidates, stats, nil
}

// estimateCandidateRecall is the paper's r-hat estimator for k-CNA quality. It
// compares candidate lists on a deterministic control subset against exact
// neighbors for those controls, avoiding an all-node exact evaluation during
// normal construction.
func estimateCandidateRecall(candidates [][]candidate, cfg candidateQualityConfig, vectors []float32, dim int, metric Metric) (float64, error) {
	if cfg.K <= 0 {
		return 0, nil
	}
	count := len(vectors) / dim
	if count == 0 {
		return 1, nil
	}
	controls := cfg.ControlIDs
	if controls == nil {
		controls = selectCandidateControls(count, cfg.Controls, cfg.Seed)
	}
	controlHits := make([]int, len(controls))
	controlTotals := make([]int, len(controls))
	_ = parallelForNodes(len(controls), cfg.Workers, func(_ int, controlIndex int) error {
		sourceID := controls[controlIndex]
		exact := exactCandidatesForNode(vectors, dim, metric, sourceID, minInt(cfg.K, count-1))
		sourceHits, sourceTotal := eval.CountHitsAtK(candidateIDs(candidates[sourceID]), candidateIDs(exact), cfg.K)
		controlHits[controlIndex] = sourceHits
		controlTotals[controlIndex] = sourceTotal
		return nil
	})

	var hits int
	var total int
	for i := range controls {
		hits += controlHits[i]
		total += controlTotals[i]
	}
	if total == 0 {
		return 1, nil
	}
	return float64(hits) / float64(total), nil
}

func selectCandidateControls(count int, maxControls int, seed int64) []int {
	if count <= 0 || maxControls == 0 {
		return nil
	}
	if maxControls >= count {
		controls := make([]int, count)
		for i := range controls {
			controls[i] = i
		}
		return controls
	}
	rng := rand.New(rand.NewSource(seed))
	perm := rng.Perm(count)
	controls := append([]int(nil), perm[:maxControls]...)
	sort.Ints(controls)
	return controls
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

// mapLayerAdjacencyToGlobal expands local IterNSG adjacency back to the full
// index shape expected by Search: adjacency[globalNodeID] -> global neighbors.
func mapLayerAdjacencyToGlobal(localAdjacency [][]int, labels []int, totalCount int) [][]int {
	globalAdjacency := make([][]int, totalCount)
	for localID, neighbors := range localAdjacency {
		globalID := labels[localID]
		globalNeighbors := make([]int, len(neighbors))
		for i, localNeighborID := range neighbors {
			globalNeighbors[i] = labels[localNeighborID]
		}
		globalAdjacency[globalID] = globalNeighbors
	}
	return globalAdjacency
}
