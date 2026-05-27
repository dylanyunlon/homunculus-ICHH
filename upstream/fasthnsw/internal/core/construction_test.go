package core

import (
	"reflect"
	"sort"
	"testing"

	"github.com/cryo-zd/fasthnsw/internal/synth"
)

func TestBuildSearchesTinyCompleteLayer(t *testing.T) {
	vectors := [][]float32{
		{0, 0},
		{1, 0},
		{2, 0},
		{3, 0},
	}
	idx, err := New(Config{Dim: 2, M: 4, K0: 4, EfConstruction: 4, Seed: 11})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build returned error: %v", err)
	}

	got, err := idx.Search([]float32{0.1, 0}, 2, 4)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	want, err := ExactTopK(idx.vectors, idx.dim, idx.cfg.Metric, []float32{0.1, 0}, 2)
	if err != nil {
		t.Fatalf("ExactTopK returned error: %v", err)
	}
	assertResults(t, got, want)
}

func TestBuildSearchesSingleVector(t *testing.T) {
	idx := mustBuildIndex(t, Config{Dim: 2}, [][]float32{{3, 4}})

	got, err := idx.Search([]float32{3, 4}, 1, 1)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	assertResults(t, got, []Result{{ID: 0, Distance: 0}})
}

func TestBuildSearchesCosineIndex(t *testing.T) {
	vectors := [][]float32{
		{1, 0},
		{0, 1},
		{1, 1},
	}
	idx, err := New(Config{Metric: MetricCosine, Dim: 2, M: 4, K0: 4, EfConstruction: 4})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build returned error: %v", err)
	}

	got, err := idx.Search([]float32{1, 0}, 2, 3)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	want, err := ExactTopK(idx.vectors, idx.dim, idx.cfg.Metric, []float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("ExactTopK returned error: %v", err)
	}
	assertResults(t, got, want)
}

func TestBuildUsesDoubleDegreeOnBaseLayer(t *testing.T) {
	idx := mustBuildIndex(t, Config{Dim: 1, M: 2, K0: 4, EfConstruction: 4, Seed: 3}, lineVectors(4))

	if len(idx.layers) == 0 {
		t.Fatal("idx.layers is empty")
	}
	for sourceID, neighbors := range idx.layers[0] {
		if len(neighbors) <= idx.cfg.M {
			t.Fatalf("base layer node %d degree = %d, want greater than M=%d in complete tiny layer", sourceID, len(neighbors), idx.cfg.M)
		}
		if len(neighbors) > baseLayerMaxDegree(idx.cfg) {
			t.Fatalf("base layer node %d degree = %d, want <= %d", sourceID, len(neighbors), baseLayerMaxDegree(idx.cfg))
		}
	}
}

func TestBuildHNSWLayerCompletesWhenDegreeEqualsBound(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(3), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}

	got, err := buildHNSWLayer(1, []int{0, 1, 2}, 3, flat, dim, Config{Metric: MetricL2}, 2)
	if err != nil {
		t.Fatalf("buildHNSWLayer returned error: %v", err)
	}
	for sourceID, neighbors := range got {
		if len(neighbors) != 2 {
			t.Fatalf("len(adjacency[%d]) = %d, want complete degree 2: %v", sourceID, len(neighbors), neighbors)
		}
	}
}

func TestBuildHNSWLayerUsesConstructionLocalVectorOrder(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{100},
		{0},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	labels := []int{1, 2, 0}
	constructionVectors := copyLayoutVectors(labels, flat, dim)

	got, err := buildHNSWLayer(0, labels, 3, constructionVectors, dim, Config{Metric: MetricL2}, 2)
	if err != nil {
		t.Fatalf("buildHNSWLayer returned error: %v", err)
	}

	assertAdjacency(t, got, [][]int{
		{2, 1},
		{2, 0},
		{1, 0},
	})
}

func TestFastHNSWOptKCNAConfigCopiesRefreshParameters(t *testing.T) {
	cfg := fastHNSWOptKCNAConfig(Config{Alpha: 90}, 12, 24, 6)

	if cfg.CandidateK != 12 {
		t.Fatalf("CandidateK = %d, want 12", cfg.CandidateK)
	}
	if cfg.SearchEf != 24 {
		t.Fatalf("SearchEf = %d, want 24", cfg.SearchEf)
	}
	if cfg.MaxDegree != 6 {
		t.Fatalf("MaxDegree = %d, want 6", cfg.MaxDegree)
	}
	if cfg.AlphaDegrees != 90 {
		t.Fatalf("AlphaDegrees = %v, want 90", cfg.AlphaDegrees)
	}
	if cfg.Workers != 0 {
		t.Fatalf("Workers = %d, want copied from zero config", cfg.Workers)
	}
}

func TestFinalHNSWLayerUsesRNGPruning(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{1, 1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := [][]candidate{
		{{id: 1}, {id: 2}},
		nil,
		nil,
	}

	finalLayer, err := buildFinalHNSWLayer(candidates, 4, flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildFinalHNSWLayer returned error: %v", err)
	}
	alphaLayer, err := buildPrunedLayer(candidates, 4, alphaPruneMode(120), flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildPrunedLayer alpha returned error: %v", err)
	}

	assertAdjacency(t, [][]int{finalLayer[0]}, [][]int{{1}})
	assertAdjacency(t, [][]int{alphaLayer[0]}, [][]int{{1, 2}})
}

func TestFinalHNSWLayerDoesNotRepairWeakComponents(t *testing.T) {
	flat, dim, err := FlattenVectors([][]float32{
		{0},
		{1},
		{10},
		{11},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := [][]candidate{
		{{id: 1}},
		{{id: 0}},
		{{id: 3}},
		{{id: 2}},
	}

	got, err := buildFinalHNSWLayer(candidates, 2, flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("buildFinalHNSWLayer returned error: %v", err)
	}

	assertAdjacency(t, got, [][]int{
		{1},
		{0},
		{3},
		{2},
	})
}

func TestBuildIsDeterministicWithFixedSeed(t *testing.T) {
	vectors := synth.UniformVectors(48, 3)
	cfg := Config{Dim: 3, M: 4, K0: 8, EfConstruction: 8, Iterations: 1, Seed: 17}

	left := mustBuildIndex(t, cfg, vectors)
	right := mustBuildIndex(t, cfg, vectors)

	if !reflect.DeepEqual(left.levels, right.levels) {
		t.Fatalf("levels differ under fixed seed:\nleft=%v\nright=%v", left.levels, right.levels)
	}
	if left.entryPoint != right.entryPoint {
		t.Fatalf("entryPoint = %d and %d, want equal", left.entryPoint, right.entryPoint)
	}
	if left.maxLayer != right.maxLayer {
		t.Fatalf("maxLayer = %d and %d, want equal", left.maxLayer, right.maxLayer)
	}
	if !reflect.DeepEqual(left.layers, right.layers) {
		t.Fatal("layers differ under fixed seed")
	}
}

func TestBuildIsDeterministicAcrossWorkerCounts(t *testing.T) {
	vectors := synth.UniformVectors(96, 4)
	cfg := Config{
		Dim:               4,
		M:                 4,
		K0:                8,
		CandidateK:        8,
		ConstructionL:     12,
		Iterations:        1,
		Seed:              29,
		CandidateRecall:   0.90,
		CandidateControls: 32,
	}

	sequential := mustBuildIndex(t, withWorkers(cfg, 1), vectors)
	parallel := mustBuildIndex(t, withWorkers(cfg, 4), vectors)

	if !reflect.DeepEqual(sequential.levels, parallel.levels) {
		t.Fatalf("levels differ across worker counts:\nworkers=1 %v\nworkers=4 %v", sequential.levels, parallel.levels)
	}
	if sequential.entryPoint != parallel.entryPoint {
		t.Fatalf("entryPoint = %d and %d, want equal", sequential.entryPoint, parallel.entryPoint)
	}
	if sequential.maxLayer != parallel.maxLayer {
		t.Fatalf("maxLayer = %d and %d, want equal", sequential.maxLayer, parallel.maxLayer)
	}
	if !reflect.DeepEqual(sequential.layers, parallel.layers) {
		t.Fatal("layers differ across worker counts")
	}

	query := []float32{0.2, 0.4, 0.6, 0.8}
	left, err := sequential.Search(query, 5, 16)
	if err != nil {
		t.Fatalf("sequential Search returned error: %v", err)
	}
	right, err := parallel.Search(query, 5, 16)
	if err != nil {
		t.Fatalf("parallel Search returned error: %v", err)
	}
	assertResults(t, right, left)
}

func TestAssignLevelsUsesSeed(t *testing.T) {
	left := assignLevels(512, 8, 1)
	right := assignLevels(512, 8, 2)
	if reflect.DeepEqual(left, right) {
		t.Fatal("assignLevels produced identical levels for different seeds")
	}
}

func TestBuildConstructionLayoutOrdersLabelsByDescendingLayer(t *testing.T) {
	levels := []int{0, 2, 1, 2, 0, 3}
	layout := buildConstructionLayout(levels, maxAssignedLayer(levels))

	assertIntSlice(t, layout.labels, []int{5, 1, 3, 2, 0, 4})
	assertIntSlice(t, layout.layerCounts, []int{6, 4, 3, 1})
}

func TestConstructionLayoutLayerPrefixesContainLayerNodeSets(t *testing.T) {
	levels := []int{1, 0, 3, 2, 1, 0, 2}
	layout := buildConstructionLayout(levels, maxAssignedLayer(levels))

	for layer := range layout.layerCounts {
		got := append([]int(nil), layout.labels[:layout.layerCounts[layer]]...)
		sort.Ints(got)
		want := nodesWithLevelAtLeast(levels, layer)
		assertIntSlice(t, got, want)
	}
}

func TestSelectEntryPointFromLayoutUsesExactTopLayer(t *testing.T) {
	levels := []int{0, 2, 1, 2, 0}
	layout := buildConstructionLayout(levels, maxAssignedLayer(levels))

	left := selectEntryPointFromLayout(layout, 2, 11)
	right := selectEntryPointFromLayout(layout, 2, 11)
	if left != right {
		t.Fatalf("entry points differ under fixed seed: %d vs %d", left, right)
	}
	if levels[left] != 2 {
		t.Fatalf("entry point level = %d, want top layer 2", levels[left])
	}
}

func TestRefineCandidatesStopsWhenQualityRequirementIsMet(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(32), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates, err := exactCandidates(flat, dim, MetricL2, 4, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}

	got, stats, err := refineCandidatesUntilRecall(candidates, optKCNAConfig{
		CandidateK:   4,
		SearchEf:     6,
		MaxDegree:    4,
		AlphaDegrees: 90,
	}, candidateQualityConfig{
		TargetRecall: 0.98,
		Controls:     16,
		Seed:         12,
		K:            4,
	}, flat, dim, MetricL2, 5)
	if err != nil {
		t.Fatalf("refineCandidatesUntilRecall returned error: %v", err)
	}
	if stats.Iterations != 0 {
		t.Fatalf("iterations = %d, want 0 when initial quality meets requirement", stats.Iterations)
	}
	if stats.InitialRecall < 0.98 || stats.FinalRecall < 0.98 {
		t.Fatalf("recall stats = %+v, want both >= 0.98", stats)
	}
	assertCandidates(t, got, candidates)
}

func TestRefineCandidatesUsesIterationsAsMaximumCap(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(48), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates := make([][]candidate, len(flat)/dim)

	_, stats, err := refineCandidatesUntilRecall(candidates, optKCNAConfig{
		CandidateK:   8,
		SearchEf:     8,
		MaxDegree:    4,
		AlphaDegrees: 90,
	}, candidateQualityConfig{
		TargetRecall: 1,
		Controls:     24,
		Seed:         19,
		K:            8,
	}, flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("refineCandidatesUntilRecall returned error: %v", err)
	}
	if stats.Iterations != 1 {
		t.Fatalf("iterations = %d, want exactly the max cap 1", stats.Iterations)
	}
}

func TestAcquireCandidatesByGraphSearchDropsSelfAndKeepsNearest(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(5), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	adjacency := completeLayer(5)

	got := acquireCandidatesByGraphSearch(adjacency, flat, dim, MetricL2, 2, 5, 1)
	assertCandidateList(t, got[2], []candidate{
		{id: 1, distance: 1},
		{id: 3, distance: 1},
	})

	parallel := acquireCandidatesByGraphSearch(adjacency, flat, dim, MetricL2, 2, 5, 4)
	assertCandidateList(t, parallel[2], got[2])
}

func TestEstimateCandidateRecallUsesFixedControls(t *testing.T) {
	flat, dim, err := FlattenVectors(lineVectors(20), 0, MetricL2)
	if err != nil {
		t.Fatalf("FlattenVectors returned error: %v", err)
	}
	candidates, err := exactCandidates(flat, dim, MetricL2, 4, 1)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}
	cfg := candidateQualityConfig{
		TargetRecall: 0.98,
		ControlIDs:   []int{1, 3, 5, 7},
		Seed:         1,
		K:            4,
	}

	left, err := estimateCandidateRecall(candidates, cfg, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("estimateCandidateRecall returned error: %v", err)
	}
	cfg.Seed = 99
	right, err := estimateCandidateRecall(candidates, cfg, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("estimateCandidateRecall second run returned error: %v", err)
	}
	if left != right {
		t.Fatalf("fixed-control recall = %v and %v, want equal", left, right)
	}
}

func withWorkers(cfg Config, workers int) Config {
	cfg.Workers = workers
	return cfg
}

func assertIntSlice(t *testing.T, got []int, want []int) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("slice = %v, want %v", got, want)
	}
}

func nodesWithLevelAtLeast(levels []int, layer int) []int {
	nodes := make([]int, 0, len(levels))
	for id, assignedLayer := range levels {
		if assignedLayer >= layer {
			nodes = append(nodes, id)
		}
	}
	return nodes
}

func TestBuildLayerInvariants(t *testing.T) {
	vectors := synth.UniformVectors(40, 2)
	cfg := Config{Dim: 2, M: 4, K0: 8, EfConstruction: 8, Iterations: 1, Seed: 23}
	idx := mustBuildIndex(t, cfg, vectors)

	if !idx.graphReady {
		t.Fatal("idx.graphReady = false, want true")
	}
	if len(idx.layers) != idx.maxLayer+1 {
		t.Fatalf("len(layers) = %d, want %d", len(idx.layers), idx.maxLayer+1)
	}
	if idx.entryPoint < 0 || idx.entryPoint >= idx.count {
		t.Fatalf("entryPoint = %d out of range", idx.entryPoint)
	}
	if idx.levels[idx.entryPoint] != idx.maxLayer {
		t.Fatalf("entryPoint level = %d, want maxLayer %d", idx.levels[idx.entryPoint], idx.maxLayer)
	}

	for layerID, layer := range idx.layers {
		if len(layer) != idx.count {
			t.Fatalf("len(layers[%d]) = %d, want %d", layerID, len(layer), idx.count)
		}
		maxDegree := idx.cfg.M
		if layerID == 0 {
			maxDegree = baseLayerMaxDegree(idx.cfg)
		}
		for sourceID, neighbors := range layer {
			if idx.levels[sourceID] < layerID && len(neighbors) != 0 {
				t.Fatalf("layer %d node %d has neighbors below assigned level: %v", layerID, sourceID, neighbors)
			}
			if len(neighbors) > maxDegree {
				t.Fatalf("layer %d node %d degree = %d, want <= %d: %v", layerID, sourceID, len(neighbors), maxDegree, neighbors)
			}
			seen := make(map[int]bool, len(neighbors))
			for _, neighborID := range neighbors {
				if neighborID == sourceID {
					t.Fatalf("layer %d node %d contains self edge: %v", layerID, sourceID, neighbors)
				}
				if neighborID < 0 || neighborID >= idx.count {
					t.Fatalf("layer %d node %d neighbor %d out of range", layerID, sourceID, neighborID)
				}
				if idx.levels[neighborID] < layerID {
					t.Fatalf("layer %d node %d links to node %d below layer", layerID, sourceID, neighborID)
				}
				if seen[neighborID] {
					t.Fatalf("layer %d node %d has duplicate neighbor %d: %v", layerID, sourceID, neighborID, neighbors)
				}
				seen[neighborID] = true
			}
		}
	}
}
