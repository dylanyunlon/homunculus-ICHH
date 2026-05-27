package fasthnsw

import (
	"io"

	"github.com/cryo-zd/fasthnsw/internal/core"
)

// Metric identifies the distance function used by an index.
type Metric = core.Metric

const (
	// MetricL2 uses squared Euclidean distance.
	MetricL2 = core.MetricL2
	// MetricCosine uses cosine distance.
	MetricCosine = core.MetricCosine
)

// Config controls index construction and search behavior.
type Config = core.Config

// Result is one nearest-neighbor result returned by Search.
type Result = core.Result

// Index stores vectors and graph metadata for approximate nearest-neighbor
// search. Its fields are intentionally private to keep the public API stable.
type Index = core.Index

// ErrNotImplemented is reserved for public API methods that may be introduced
// before their full implementation is available.
var ErrNotImplemented = core.ErrNotImplemented

// ErrIndexNotBuilt is returned by Search when an index has vectors but no
// searchable graph metadata yet.
var ErrIndexNotBuilt = core.ErrIndexNotBuilt

// DefaultConfig returns deterministic defaults suitable for a first index.
func DefaultConfig() Config {
	return core.DefaultConfig()
}

// New creates an index with validated configuration.
func New(cfg Config) (*Index, error) {
	return core.New(cfg)
}

// Load reads a versioned binary index representation written by Save.
func Load(r io.Reader) (*Index, error) {
	return core.Load(r)
}
