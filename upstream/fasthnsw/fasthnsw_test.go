package fasthnsw

import (
	"bytes"
	"testing"
)

func TestFacadePublicAPI(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Dim = 2
	cfg.M = 2
	cfg.K0 = 2
	cfg.CandidateK = 2
	cfg.ConstructionL = 2

	idx, err := New(cfg)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build([][]float32{{0, 0}, {1, 0}, {0, 1}}); err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	results, err := idx.Search([]float32{0, 0}, 1, 2)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) != 1 || results[0].ID != 0 {
		t.Fatalf("Search results = %v, want id 0", results)
	}

	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	loaded, err := Load(&buf)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	results, err = loaded.Search([]float32{0, 0}, 1, 2)
	if err != nil {
		t.Fatalf("loaded Search returned error: %v", err)
	}
	if len(results) != 1 || results[0].ID != 0 {
		t.Fatalf("loaded Search results = %v, want id 0", results)
	}
}
