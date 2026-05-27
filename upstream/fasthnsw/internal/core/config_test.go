package core

import (
	"math"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Metric != MetricL2 {
		t.Fatalf("default metric = %v, want %v", cfg.Metric, MetricL2)
	}
	if cfg.M != defaultM {
		t.Fatalf("default M = %d, want %d", cfg.M, defaultM)
	}
	if cfg.EfConstruction != defaultEfConstruction {
		t.Fatalf("default EfConstruction = %d, want %d", cfg.EfConstruction, defaultEfConstruction)
	}
	if cfg.K0 != defaultK0 {
		t.Fatalf("default K0 = %d, want %d", cfg.K0, defaultK0)
	}
	if cfg.CandidateK != defaultCandidateK {
		t.Fatalf("default CandidateK = %d, want %d", cfg.CandidateK, defaultCandidateK)
	}
	if cfg.ConstructionL != defaultConstructionL {
		t.Fatalf("default ConstructionL = %d, want %d", cfg.ConstructionL, defaultConstructionL)
	}
	if cfg.Alpha != defaultAlpha {
		t.Fatalf("default Alpha = %v, want %v", cfg.Alpha, defaultAlpha)
	}
	if cfg.Iterations != defaultIterations {
		t.Fatalf("default Iterations = %d, want %d", cfg.Iterations, defaultIterations)
	}
	if cfg.Seed != defaultSeed {
		t.Fatalf("default Seed = %d, want %d", cfg.Seed, defaultSeed)
	}
	if cfg.Workers <= 0 {
		t.Fatalf("default Workers = %d, want positive", cfg.Workers)
	}
	if cfg.CandidateRecall != defaultCandidateRecall {
		t.Fatalf("default CandidateRecall = %v, want %v", cfg.CandidateRecall, defaultCandidateRecall)
	}
	if cfg.CandidateControls != defaultCandidateControls {
		t.Fatalf("default CandidateControls = %d, want %d", cfg.CandidateControls, defaultCandidateControls)
	}
}

func TestNewAcceptsDefaultConfig(t *testing.T) {
	idx, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New(DefaultConfig()) returned error: %v", err)
	}
	if idx == nil {
		t.Fatal("New(DefaultConfig()) returned nil index")
	}
}

func TestNewNormalizesZeroConfig(t *testing.T) {
	idx, err := New(Config{})
	if err != nil {
		t.Fatalf("New(Config{}) returned error: %v", err)
	}
	if idx.cfg.M != defaultM {
		t.Fatalf("normalized M = %d, want %d", idx.cfg.M, defaultM)
	}
	if idx.cfg.Workers <= 0 {
		t.Fatalf("normalized Workers = %d, want positive", idx.cfg.Workers)
	}
	if idx.cfg.CandidateK != idx.cfg.ConstructionL {
		t.Fatalf("normalized CandidateK = %d, want ConstructionL %d", idx.cfg.CandidateK, idx.cfg.ConstructionL)
	}
}

func TestNewUsesEfConstructionAsConstructionLAlias(t *testing.T) {
	idx, err := New(Config{EfConstruction: 64})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if idx.cfg.ConstructionL != 64 {
		t.Fatalf("ConstructionL = %d, want EfConstruction alias 64", idx.cfg.ConstructionL)
	}
	if idx.cfg.CandidateK != 64 {
		t.Fatalf("CandidateK = %d, want default to ConstructionL 64", idx.cfg.CandidateK)
	}
}

func TestNewSeparatesCandidateKAndConstructionL(t *testing.T) {
	idx, err := New(Config{CandidateK: 16, ConstructionL: 64})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if idx.cfg.CandidateK != 16 {
		t.Fatalf("CandidateK = %d, want 16", idx.cfg.CandidateK)
	}
	if idx.cfg.ConstructionL != 64 {
		t.Fatalf("ConstructionL = %d, want 64", idx.cfg.ConstructionL)
	}
}

func TestNewRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "metric", cfg: Config{Metric: Metric(99)}},
		{name: "dim", cfg: Config{Dim: -1}},
		{name: "m", cfg: Config{M: -1}},
		{name: "ef construction", cfg: Config{EfConstruction: -1}},
		{name: "k0", cfg: Config{K0: -1}},
		{name: "candidate k", cfg: Config{CandidateK: -1}},
		{name: "construction l", cfg: Config{ConstructionL: -1}},
		{name: "construction l less than candidate k", cfg: Config{CandidateK: 16, ConstructionL: 8}},
		{name: "alpha", cfg: Config{Alpha: 59}},
		{name: "alpha high", cfg: Config{Alpha: 181}},
		{name: "alpha nan", cfg: Config{Alpha: math.NaN()}},
		{name: "iterations", cfg: Config{Iterations: -1}},
		{name: "workers", cfg: Config{Workers: -1}},
		{name: "candidate recall low", cfg: Config{CandidateRecall: -1}},
		{name: "candidate recall high", cfg: Config{CandidateRecall: 1.01}},
		{name: "candidate recall nan", cfg: Config{CandidateRecall: math.NaN()}},
		{name: "candidate controls", cfg: Config{CandidateControls: -1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := New(tt.cfg); err == nil {
				t.Fatal("New returned nil error")
			}
		})
	}
}
