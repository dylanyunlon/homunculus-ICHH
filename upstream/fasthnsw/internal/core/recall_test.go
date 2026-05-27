package core

import (
	"testing"

	"github.com/cryo-zd/fasthnsw/internal/eval"
	"github.com/cryo-zd/fasthnsw/internal/synth"
)

func TestBuildSearchRecallClusteredData(t *testing.T) {
	vectors := synth.ClusteredVectors(180, 6, 6)
	queries := synth.ClusteredQueries(36, 6, 6)
	idx := mustBuildIndex(t, recallTestConfig(6), vectors)

	recall := averageRecallAtK(t, idx, queries, 10, 48)
	if recall < 0.85 {
		t.Fatalf("clustered Recall@10 = %.3f, want >= 0.85", recall)
	}
}

func TestBuildSearchRecallUniformData(t *testing.T) {
	vectors := synth.UniformVectors(160, 5)
	queries := synth.UniformQueries(32, 5)
	idx := mustBuildIndex(t, recallTestConfig(5), vectors)

	recall := averageRecallAtK(t, idx, queries, 10, 64)
	if recall < 0.70 {
		t.Fatalf("uniform Recall@10 = %.3f, want >= 0.70", recall)
	}
}

func TestBuildSearchResultsDeterministicWithFixedSeed(t *testing.T) {
	vectors := synth.ClusteredVectors(120, 4, 4)
	queries := synth.ClusteredQueries(8, 4, 4)
	cfg := recallTestConfig(4)

	left := mustBuildIndex(t, cfg, vectors)
	right := mustBuildIndex(t, cfg, vectors)

	for _, query := range queries {
		leftResults, err := left.Search(query, 10, 48)
		if err != nil {
			t.Fatalf("left Search returned error: %v", err)
		}
		rightResults, err := right.Search(query, 10, 48)
		if err != nil {
			t.Fatalf("right Search returned error: %v", err)
		}
		assertResults(t, leftResults, rightResults)
	}
}

func recallTestConfig(dim int) Config {
	return Config{
		Dim:               dim,
		M:                 12,
		K0:                32,
		CandidateK:        32,
		ConstructionL:     64,
		Alpha:             67,
		Iterations:        3,
		Seed:              41,
		CandidateRecall:   0.90,
		CandidateControls: 64,
	}
}

func averageRecallAtK(t *testing.T, idx *Index, queries [][]float32, k int, efSearch int) float64 {
	t.Helper()

	var total float64
	for _, query := range queries {
		got, err := idx.Search(query, k, efSearch)
		if err != nil {
			t.Fatalf("Search returned error: %v", err)
		}
		want, err := ExactTopK(idx.vectors, idx.dim, idx.cfg.Metric, query, k)
		if err != nil {
			t.Fatalf("ExactTopK returned error: %v", err)
		}
		total += eval.RecallAtK(ResultIDs(got, k), ResultIDs(want, k), k)
	}
	return total / float64(len(queries))
}
