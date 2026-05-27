package main

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/cryo-zd/fasthnsw"
)

func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	return fs
}

type configFlags struct {
	metric            string
	dim               int
	m                 int
	k0                int
	candidateK        int
	constructionL     int
	alpha             float64
	iterations        int
	seed              int64
	workers           int
	candidateRecall   float64
	candidateControls int
}

func defaultConfigFlags() configFlags {
	cfg := fasthnsw.DefaultConfig()
	return configFlags{
		metric:            "l2",
		dim:               0,
		m:                 cfg.M,
		k0:                cfg.K0,
		candidateK:        cfg.CandidateK,
		constructionL:     cfg.ConstructionL,
		alpha:             cfg.Alpha,
		iterations:        cfg.Iterations,
		seed:              cfg.Seed,
		workers:           cfg.Workers,
		candidateRecall:   cfg.CandidateRecall,
		candidateControls: cfg.CandidateControls,
	}
}

func addConfigFlags(fs *flag.FlagSet, cfg *configFlags) {
	fs.StringVar(&cfg.metric, "metric", cfg.metric, "distance metric: l2 or cosine")
	fs.IntVar(&cfg.dim, "dim", cfg.dim, "vector dimension; zero infers from input")
	fs.IntVar(&cfg.m, "m", cfg.m, "HNSW max degree parameter")
	fs.IntVar(&cfg.k0, "k0", cfg.k0, "initial approximate KNNG size")
	fs.IntVar(&cfg.candidateK, "candidate-k", cfg.candidateK, "retained k-CNA candidate count")
	fs.IntVar(&cfg.constructionL, "construction-l", cfg.constructionL, "construction graph-search width")
	fs.Float64Var(&cfg.alpha, "alpha", cfg.alpha, "alpha-pruning threshold in degrees")
	fs.IntVar(&cfg.iterations, "iterations", cfg.iterations, "maximum IterNSG refinement iterations")
	fs.Int64Var(&cfg.seed, "seed", cfg.seed, "deterministic construction seed")
	fs.IntVar(&cfg.workers, "workers", cfg.workers, "build-time worker count; zero uses the library default")
	fs.Float64Var(&cfg.candidateRecall, "candidate-recall", cfg.candidateRecall, "IterNSG candidate recall requirement")
	fs.IntVar(&cfg.candidateControls, "candidate-controls", cfg.candidateControls, "deterministic candidate-quality estimator sample size")
}

func (cfg configFlags) config() (fasthnsw.Config, error) {
	metric, err := parseMetric(cfg.metric)
	if err != nil {
		return fasthnsw.Config{}, err
	}
	return fasthnsw.Config{
		Metric:            metric,
		Dim:               cfg.dim,
		M:                 cfg.m,
		K0:                cfg.k0,
		CandidateK:        cfg.candidateK,
		ConstructionL:     cfg.constructionL,
		Alpha:             cfg.alpha,
		Iterations:        cfg.iterations,
		Seed:              cfg.seed,
		Workers:           cfg.workers,
		CandidateRecall:   cfg.candidateRecall,
		CandidateControls: cfg.candidateControls,
	}, nil
}

func parseMetric(value string) (fasthnsw.Metric, error) {
	switch strings.ToLower(value) {
	case "l2":
		return fasthnsw.MetricL2, nil
	case "cosine":
		return fasthnsw.MetricCosine, nil
	default:
		return 0, fmt.Errorf("unsupported metric %q", value)
	}
}

func metricName(metric fasthnsw.Metric) string {
	switch metric {
	case fasthnsw.MetricL2:
		return "l2"
	case fasthnsw.MetricCosine:
		return "cosine"
	default:
		return fmt.Sprintf("metric-%d", metric)
	}
}
