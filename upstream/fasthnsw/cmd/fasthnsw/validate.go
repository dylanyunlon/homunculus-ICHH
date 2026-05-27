package main

import (
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/cryo-zd/fasthnsw"
	"github.com/cryo-zd/fasthnsw/internal/benchdata"
)

func runValidate(args []string, stdout io.Writer) error {
	cfgFlags := defaultConfigFlags()
	fs := newFlagSet("validate")
	algorithm := fs.String("algorithm", string(benchdata.AlgorithmFastHNSW), "construction algorithm: fasthnsw or hnsw")
	datasetKind := fs.String("dataset", "clustered", "dataset type: clustered, uniform, hdf5, fvecs, or bvecs")
	inputPath := fs.String("input", "", "ANN-Benchmarks HDF5 input file")
	basePath := fs.String("base", "", "raw base vector file")
	queriesPath := fs.String("queries", "", "raw query vector file")
	truthPath := fs.String("truth", "", "raw ground-truth ivecs file")
	vectorCount := fs.Int("vectors", 180, "synthetic base vector count")
	queryCount := fs.Int("query-count", 36, "synthetic query vector count")
	clusters := fs.Int("clusters", 6, "clustered synthetic dataset cluster count")
	limitBase := fs.Int("limit-base", 0, "maximum base vectors to load; zero loads all")
	limitQueries := fs.Int("limit-queries", 0, "maximum queries to load; zero loads all")
	k := fs.Int("k", 10, "number of nearest neighbors")
	efSearch := fs.Int("ef", 64, "HNSW search width")
	saveIndexPath := fs.String("save-index", "", "optional path for saving the built index")
	addConfigFlags(fs, &cfgFlags)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("validate received unexpected positional arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *k <= 0 {
		return errors.New("validate requires positive -k")
	}
	if *efSearch < *k {
		return errors.New("validate requires -ef greater than or equal to -k")
	}

	cfg, err := cfgFlags.config()
	if err != nil {
		return err
	}
	dataset, err := loadValidationDataset(*datasetKind, cfg.Metric, *inputPath, *basePath, *queriesPath, *truthPath, *vectorCount, *queryCount, validationDim(cfg.Dim), *clusters, *limitBase, *limitQueries, *k)
	if err != nil {
		return err
	}

	validationAlgorithm, err := parseValidationAlgorithm(*algorithm)
	if err != nil {
		return err
	}
	cfg = displayConfigForAlgorithm(cfg, validationAlgorithm)
	result, err := benchdata.RunValidationWithAlgorithm(dataset, cfg, validationAlgorithm, *k, *efSearch, *saveIndexPath)
	if err != nil {
		return err
	}
	printValidationResult(stdout, cfg, result)
	return nil
}

func parseValidationAlgorithm(value string) (benchdata.ValidationAlgorithm, error) {
	switch value {
	case string(benchdata.AlgorithmFastHNSW):
		return benchdata.AlgorithmFastHNSW, nil
	case string(benchdata.AlgorithmHNSW):
		return benchdata.AlgorithmHNSW, nil
	default:
		return "", fmt.Errorf("unsupported validation algorithm %q", value)
	}
}

func displayConfigForAlgorithm(cfg fasthnsw.Config, algorithm benchdata.ValidationAlgorithm) fasthnsw.Config {
	if algorithm != benchdata.AlgorithmHNSW {
		return cfg
	}
	if cfg.ConstructionL == 0 {
		cfg.ConstructionL = cfg.EfConstruction
	}
	if cfg.ConstructionL > 0 && cfg.CandidateK > cfg.ConstructionL {
		cfg.CandidateK = cfg.ConstructionL
	}
	return cfg
}

func loadValidationDataset(kind string, metric fasthnsw.Metric, inputPath string, basePath string, queriesPath string, truthPath string, vectorCount int, queryCount int, dim int, clusters int, limitBase int, limitQueries int, k int) (benchdata.Dataset, error) {
	switch kind {
	case "uniform", "clustered":
		return benchdata.LoadSynthetic(benchdata.SyntheticOptions{
			Shape:    kind,
			Vectors:  vectorCount,
			Queries:  queryCount,
			Dim:      dim,
			Clusters: clusters,
			Metric:   metric,
			K:        k,
		})
	case "hdf5":
		if inputPath == "" {
			return benchdata.Dataset{}, errors.New("hdf5 validation requires -input")
		}
		return benchdata.LoadANNBenchHDF5(inputPath, metric, limitBase, limitQueries)
	case "fvecs", "bvecs":
		return benchdata.LoadRawDataset(kind, basePath, queriesPath, truthPath, metric, limitBase, limitQueries)
	default:
		return benchdata.Dataset{}, fmt.Errorf("unsupported validation dataset %q", kind)
	}
}

func validationDim(configuredDim int) int {
	if configuredDim > 0 {
		return configuredDim
	}
	return 8
}

func printValidationResult(stdout io.Writer, cfg fasthnsw.Config, result benchdata.ValidationResult) {
	fmt.Fprintf(stdout, "algorithm=%s\n", result.Algorithm)
	fmt.Fprintf(stdout, "dataset=%s\n", result.Dataset.Name)
	fmt.Fprintf(stdout, "metric=%s\n", metricName(result.Dataset.Metric))
	fmt.Fprintf(stdout, "vectors=%d\n", len(result.Dataset.Base))
	fmt.Fprintf(stdout, "queries=%d\n", len(result.Dataset.Queries))
	fmt.Fprintf(stdout, "dim=%d\n", result.Dataset.Dim())
	fmt.Fprintf(stdout, "m=%d\n", cfg.M)
	if result.Algorithm == benchdata.AlgorithmHNSW {
		fmt.Fprintf(stdout, "ef_construction=%d\n", cfg.ConstructionL)
	} else {
		fmt.Fprintf(stdout, "k0=%d\n", cfg.K0)
		fmt.Fprintf(stdout, "candidate_k=%d\n", cfg.CandidateK)
		fmt.Fprintf(stdout, "construction_l=%d\n", cfg.ConstructionL)
		fmt.Fprintf(stdout, "alpha=%.6f\n", cfg.Alpha)
		fmt.Fprintf(stdout, "iterations=%d\n", cfg.Iterations)
	}
	fmt.Fprintf(stdout, "seed=%d\n", cfg.Seed)
	fmt.Fprintf(stdout, "workers=%d\n", cfg.Workers)
	fmt.Fprintf(stdout, "k=%d\n", result.K)
	fmt.Fprintf(stdout, "ef=%d\n", result.EfSearch)
	fmt.Fprintf(stdout, "build_s=%.3f\n", float64(result.BuildTime.Seconds()))
	fmt.Fprintf(stdout, "query_s=%.3f\n", float64(result.QueryTime.Seconds()))
	fmt.Fprintf(stdout, "qps=%.3f\n", result.QPS())
	fmt.Fprintf(stdout, "recall_at_%d=%.6f\n", result.K, result.Recall)
	if result.IndexBytes > 0 {
		fmt.Fprintf(stdout, "index_bytes=%d\n", result.IndexBytes)
	}
}
