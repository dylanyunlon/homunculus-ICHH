package benchdata

import (
	"fmt"

	"github.com/cryo-zd/fasthnsw"
	"github.com/cryo-zd/fasthnsw/internal/synth"
)

// SyntheticOptions controls deterministic generated validation datasets.
type SyntheticOptions struct {
	Shape    string
	Vectors  int
	Queries  int
	Dim      int
	Clusters int
	Metric   fasthnsw.Metric
	K        int
}

// LoadSynthetic returns a deterministic generated dataset with exact ground
// truth. It is useful for local smoke validation when public datasets are not
// present on the machine.
func LoadSynthetic(opts SyntheticOptions) (Dataset, error) {
	if opts.Shape == "" {
		opts.Shape = "clustered"
	}
	if opts.Vectors <= 0 {
		return Dataset{}, fmt.Errorf("synthetic vectors must be positive")
	}
	if opts.Queries <= 0 {
		return Dataset{}, fmt.Errorf("synthetic queries must be positive")
	}
	if opts.Dim <= 0 {
		return Dataset{}, fmt.Errorf("synthetic dimension must be positive")
	}
	if opts.K <= 0 {
		return Dataset{}, fmt.Errorf("k must be positive")
	}

	var base [][]float32
	var queries [][]float32
	name := opts.Shape
	switch opts.Shape {
	case "uniform":
		base = synth.UniformVectors(opts.Vectors, opts.Dim)
		queries = synth.UniformQueries(opts.Queries, opts.Dim)
	case "clustered":
		if opts.Clusters <= 0 {
			return Dataset{}, fmt.Errorf("synthetic clusters must be positive")
		}
		base = synth.ClusteredVectors(opts.Vectors, opts.Dim, opts.Clusters)
		queries = synth.ClusteredQueries(opts.Queries, opts.Dim, opts.Clusters)
	default:
		return Dataset{}, fmt.Errorf("unsupported generated dataset shape %q", opts.Shape)
	}

	truth, err := ExactGroundTruth(base, queries, opts.Metric, opts.K)
	if err != nil {
		return Dataset{}, err
	}
	return Dataset{
		Name:        name,
		Metric:      opts.Metric,
		Base:        base,
		Queries:     queries,
		GroundTruth: truth,
	}, nil
}
