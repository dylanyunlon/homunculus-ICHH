package core

import (
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/cryo-zd/fasthnsw/internal/eval"
	"github.com/cryo-zd/fasthnsw/internal/synth"
)

var benchmarkIndexSink *Index

func BenchmarkDistanceL2(b *testing.B) {
	left := synth.UniformVectors(1, 128)[0]
	right := synth.UniformQueries(1, 128)[0]

	b.ReportAllocs()
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		sink += squaredL2(left, right)
	}
	if sink == 0 {
		b.Fatal("distance sink was zero")
	}
}

func BenchmarkDistanceCosine(b *testing.B) {
	left := make([]float32, 128)
	right := make([]float32, 128)
	if err := normalizeInto(left, synth.UniformVectors(1, 128)[0]); err != nil {
		b.Fatalf("normalize left: %v", err)
	}
	if err := normalizeInto(right, synth.UniformQueries(1, 128)[0]); err != nil {
		b.Fatalf("normalize right: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		sink += cosineDistanceNormalized(left, right)
	}
	if sink == 0 {
		b.Fatal("distance sink was zero")
	}
}

func BenchmarkCandidateAcquisition(b *testing.B) {
	vectors, dim, err := FlattenVectors(synth.UniformVectors(256, 8), 0, MetricL2)
	if err != nil {
		b.Fatalf("FlattenVectors returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := approximateKNNGCandidates(vectors, dim, MetricL2, 16, int64(i+1), 4, 1); err != nil {
			b.Fatalf("approximateKNNGCandidates returned error: %v", err)
		}
	}
}

func BenchmarkLayerConstruction(b *testing.B) {
	vectors, dim, err := FlattenVectors(synth.UniformVectors(256, 8), 0, MetricL2)
	if err != nil {
		b.Fatalf("FlattenVectors returned error: %v", err)
	}
	nodes := make([]int, 256)
	for i := range nodes {
		nodes[i] = i
	}
	cfg := Config{
		Metric:            MetricL2,
		M:                 8,
		K0:                16,
		CandidateK:        16,
		ConstructionL:     32,
		Alpha:             67,
		Iterations:        2,
		Seed:              99,
		CandidateRecall:   0.90,
		CandidateControls: 64,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := buildHNSWLayer(0, nodes, len(nodes), vectors, dim, cfg, baseLayerMaxDegree(cfg)); err != nil {
			b.Fatalf("buildHNSWLayer returned error: %v", err)
		}
	}
}

func BenchmarkBuild(b *testing.B) {
	vectors := synth.UniformVectors(256, 8)
	cfg := Config{Dim: 8, M: 8, K0: 16, CandidateK: 16, ConstructionL: 32, Iterations: 2, Seed: 99, CandidateRecall: 0.90, CandidateControls: 64}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := New(cfg)
		if err != nil {
			b.Fatalf("New returned error: %v", err)
		}
		if err := idx.Build(vectors); err != nil {
			b.Fatalf("Build returned error: %v", err)
		}
	}
}

func BenchmarkBuildAlgorithms(b *testing.B) {
	vectors := synth.ClusteredVectors(1024, 16, 32)
	queries := synth.ClusteredQueries(16, 16, 32)
	cfg := Config{Dim: 16, M: 12, K0: 48, CandidateK: 48, ConstructionL: 48, Iterations: 1, Seed: 101, CandidateRecall: 0.90, CandidateControls: 128, Workers: 1}

	b.Run("fasthnsw", func(b *testing.B) {
		benchmarkBuildFactory(b, queries, func() (*Index, error) {
			idx, err := New(cfg)
			if err != nil {
				return nil, err
			}
			if err := idx.Build(vectors); err != nil {
				return nil, err
			}
			return idx, nil
		})
	})
	b.Run("hnsw", func(b *testing.B) {
		benchmarkBuildFactory(b, queries, func() (*Index, error) {
			return BuildStandardHNSWForBenchmark(cfg, vectors)
		})
	})
}

func BenchmarkBuildWorkers(b *testing.B) {
	vectors := synth.UniformVectors(1024, 16)
	baseCfg := Config{Dim: 16, M: 12, K0: 24, CandidateK: 24, ConstructionL: 48, Iterations: 2, Seed: 101, CandidateRecall: 0.90, CandidateControls: 128}

	for _, tt := range []struct {
		name    string
		workers int
	}{
		{name: "workers1", workers: 1},
		{name: "workers4", workers: 4},
		{name: "workersDefault", workers: runtime.GOMAXPROCS(0)},
	} {
		b.Run(tt.name, func(b *testing.B) {
			cfg := baseCfg
			cfg.Workers = tt.workers

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx, err := New(cfg)
				if err != nil {
					b.Fatalf("New returned error: %v", err)
				}
				if err := idx.Build(vectors); err != nil {
					b.Fatalf("Build returned error: %v", err)
				}
			}
		})
	}
}

func BenchmarkBuildScale(b *testing.B) {
	for _, shape := range constructionBenchmarkShapes() {
		for _, workerCount := range constructionBenchmarkWorkers() {
			b.Run(fmt.Sprintf("%s/workers%d", shape.name, workerCount), func(b *testing.B) {
				vectors, queries := shape.dataset()
				cfg := constructionBenchmarkConfig(shape.dim, workerCount)
				benchmarkBuildFactory(b, queries, func() (*Index, error) {
					idx, err := New(cfg)
					if err != nil {
						return nil, err
					}
					if err := idx.Build(vectors); err != nil {
						return nil, err
					}
					return idx, nil
				})
			})
		}
	}
}

func BenchmarkBuildProfile(b *testing.B) {
	vectors := synth.UniformVectors(8192, 32)
	for _, workerCount := range []int{1, 4} {
		b.Run(fmt.Sprintf("uniform_8192x32/workers%d", workerCount), func(b *testing.B) {
			cfg := constructionBenchmarkConfig(32, workerCount)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx, err := New(cfg)
				if err != nil {
					b.Fatalf("New returned error: %v", err)
				}
				if err := idx.Build(vectors); err != nil {
					b.Fatalf("Build returned error: %v", err)
				}
				benchmarkIndexSink = idx
			}
		})
	}
}

func BenchmarkSearch(b *testing.B) {
	vectors := synth.UniformVectors(256, 8)
	idx, err := New(Config{Dim: 8, M: 8, K0: 16, CandidateK: 16, ConstructionL: 32, Iterations: 2, Seed: 99, CandidateRecall: 0.90, CandidateControls: 64})
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		b.Fatalf("Build returned error: %v", err)
	}
	query := synth.UniformQueries(1, 8)[0]

	for _, efSearch := range []int{16, 32, 64} {
		b.Run(fmt.Sprintf("ef%d", efSearch), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := idx.Search(query, 10, efSearch); err != nil {
					b.Fatalf("Search returned error: %v", err)
				}
			}
		})
	}
}

type benchmarkDatasetKind int

const (
	benchmarkDatasetUniform benchmarkDatasetKind = iota
	benchmarkDatasetClustered
)

type constructionBenchmarkShape struct {
	name        string
	count       int
	dim         int
	clusters    int
	datasetKind benchmarkDatasetKind
}

func constructionBenchmarkShapes() []constructionBenchmarkShape {
	shapes := []constructionBenchmarkShape{
		{name: "uniform_8192x32", count: 8192, dim: 32, datasetKind: benchmarkDatasetUniform},
		{name: "clustered_8192x32", count: 8192, dim: 32, clusters: 64, datasetKind: benchmarkDatasetClustered},
	}
	if os.Getenv("FASTHNSW_LARGE_BENCH") == "1" {
		shapes = append(shapes,
			constructionBenchmarkShape{name: "uniform_32768x32", count: 32768, dim: 32, datasetKind: benchmarkDatasetUniform},
			constructionBenchmarkShape{name: "clustered_32768x32", count: 32768, dim: 32, clusters: 128, datasetKind: benchmarkDatasetClustered},
		)
	}
	return shapes
}

func (shape constructionBenchmarkShape) dataset() ([][]float32, [][]float32) {
	const queryCount = 16
	switch shape.datasetKind {
	case benchmarkDatasetUniform:
		return synth.UniformVectors(shape.count, shape.dim), synth.UniformQueries(queryCount, shape.dim)
	case benchmarkDatasetClustered:
		return synth.ClusteredVectors(shape.count, shape.dim, shape.clusters), synth.ClusteredQueries(queryCount, shape.dim, shape.clusters)
	default:
		panic("fasthnsw: unsupported benchmark dataset kind")
	}
}

func constructionBenchmarkWorkers() []int {
	defaultWorkers := runtime.GOMAXPROCS(0)
	if defaultWorkers == 4 {
		return []int{1, 2, 4}
	}
	return []int{1, 2, 4, defaultWorkers}
}

func constructionBenchmarkConfig(dim int, workers int) Config {
	return Config{
		Dim:               dim,
		M:                 12,
		K0:                24,
		CandidateK:        24,
		ConstructionL:     48,
		Iterations:        2,
		Seed:              101,
		CandidateRecall:   0.90,
		CandidateControls: 128,
		Workers:           workers,
	}
}

func benchmarkBuildFactory(b *testing.B, queries [][]float32, build func() (*Index, error)) {
	b.Helper()
	recall, qps := measureConstructionQuality(b, build, queries, 10, 64)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := build()
		if err != nil {
			b.Fatalf("build returned error: %v", err)
		}
		benchmarkIndexSink = idx
	}
	b.StopTimer()
	b.ReportMetric(recall, "recall@10")
	b.ReportMetric(qps, "qps")
}

func measureConstructionQuality(b *testing.B, build func() (*Index, error), queries [][]float32, k int, efSearch int) (float64, float64) {
	b.Helper()
	idx, err := build()
	if err != nil {
		b.Fatalf("quality build returned error: %v", err)
	}

	var hits int
	var total int
	for _, query := range queries {
		exact, err := ExactTopK(idx.vectors, idx.dim, idx.cfg.Metric, query, k)
		if err != nil {
			b.Fatalf("ExactTopK returned error: %v", err)
		}
		got, err := idx.Search(query, k, efSearch)
		if err != nil {
			b.Fatalf("Search returned error: %v", err)
		}
		queryHits, queryTotal := eval.CountHitsAtK(ResultIDs(got, k), ResultIDs(exact, k), k)
		hits += queryHits
		total += queryTotal
	}
	recall := 0.0
	if total > 0 {
		recall = float64(hits) / float64(total)
	}

	const searchRepeats = 4
	start := time.Now()
	var searches int
	for repeat := 0; repeat < searchRepeats; repeat++ {
		for _, query := range queries {
			if _, err := idx.Search(query, k, efSearch); err != nil {
				b.Fatalf("Search returned error during QPS report: %v", err)
			}
			searches++
		}
	}
	elapsed := time.Since(start)
	qps := 0.0
	if elapsed > 0 {
		qps = float64(searches) / elapsed.Seconds()
	}
	return recall, qps
}
