package core

import (
	"bytes"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/cryo-zd/fasthnsw/internal/synth"
)

func TestBuildStoresVectorsAndConstructsGraph(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	if idx.dim != 2 {
		t.Fatalf("idx.dim = %d, want 2", idx.dim)
	}
	if idx.count != 2 {
		t.Fatalf("idx.count = %d, want 2", idx.count)
	}
	if len(idx.vectors) != 4 {
		t.Fatalf("len(idx.vectors) = %d, want 4", len(idx.vectors))
	}
	if idx.vectors[0] != 1 || idx.vectors[3] != 4 {
		t.Fatalf("stored vectors = %v, want flattened input", idx.vectors)
	}
	if !idx.graphReady {
		t.Fatal("idx.graphReady = false, want true")
	}
	if idx.entryPoint < 0 {
		t.Fatalf("idx.entryPoint = %d, want non-negative", idx.entryPoint)
	}
	if len(idx.layers) == 0 {
		t.Fatal("idx.layers is empty")
	}
	if len(idx.levels) != idx.count {
		t.Fatalf("len(idx.levels) = %d, want %d", len(idx.levels), idx.count)
	}
}

func TestBuildInfersDimension(t *testing.T) {
	idx, err := New(Config{})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{1, 2, 3}, {4, 5, 6}})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	if idx.dim != 3 {
		t.Fatalf("idx.dim = %d, want 3", idx.dim)
	}
	if idx.cfg.Dim != 3 {
		t.Fatalf("idx.cfg.Dim = %d, want 3", idx.cfg.Dim)
	}
}

func TestBuildStoresNormalizedCosineVectors(t *testing.T) {
	idx, err := New(Config{Metric: MetricCosine})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{3, 4}})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	if !almostEqual(idx.vectors[0], 0.6) || !almostEqual(idx.vectors[1], 0.8) {
		t.Fatalf("stored cosine vector = %v, want normalized [0.6 0.8]", idx.vectors)
	}
}

func TestBuildRejectsInvalidVectors(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tests := []struct {
		name    string
		vectors [][]float32
	}{
		{name: "empty", vectors: nil},
		{name: "zero dim", vectors: [][]float32{{}}},
		{name: "mismatch", vectors: [][]float32{{1, 2}, {3}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := idx.Build(tt.vectors)
			if err == nil {
				t.Fatal("Build returned nil error")
			}
		})
	}
}

func TestBuildRejectsCosineZeroVector(t *testing.T) {
	idx, err := New(Config{Metric: MetricCosine, Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{1, 0}, {0, 0}})
	if err == nil {
		t.Fatal("Build returned nil error")
	}
}

func TestSearchValidatesInputsBeforeGraphCheck(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tests := []struct {
		name     string
		query    []float32
		k        int
		efSearch int
	}{
		{name: "empty query", query: nil, k: 1, efSearch: 2},
		{name: "wrong dimension", query: []float32{1}, k: 1, efSearch: 2},
		{name: "bad k", query: []float32{1, 2}, k: 0, efSearch: 2},
		{name: "bad efSearch", query: []float32{1, 2}, k: 1, efSearch: 0},
		{name: "efSearch less than k", query: []float32{1, 2}, k: 2, efSearch: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := idx.Search(tt.query, tt.k, tt.efSearch)
			if err == nil {
				t.Fatal("Search returned nil error")
			}
		})
	}
}

func TestSearchUnbuiltIndexReturnsNotBuilt(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = idx.Search([]float32{1, 2}, 1, 1)
	if !errors.Is(err, ErrIndexNotBuilt) {
		t.Fatalf("Search error = %v, want ErrIndexNotBuilt", err)
	}
}

func TestSearchTrustsReadyGraphInHotPath(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	err = idx.Build([][]float32{{0, 0}, {1, 0}})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}

	idx.layers = [][][]int{{{1}}}
	idx.entryPoint = 0
	idx.maxLayer = 0
	idx.graphReady = true

	got, err := idx.Search([]float32{0, 0}, 1, 1)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	assertResults(t, got, []Result{{ID: 0, Distance: 0}})
}

func TestSaveRejectsUnbuiltIndex(t *testing.T) {
	idx, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if err := idx.Save(bytes.NewBuffer(nil)); !errors.Is(err, ErrIndexNotBuilt) {
		t.Fatalf("Save error = %v, want ErrIndexNotBuilt", err)
	}
}

func TestSaveAndLoadRejectNilIO(t *testing.T) {
	idx, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if err := idx.Save(nil); err == nil {
		t.Fatalf("Save(nil) error = %v, want validation error", err)
	}
	if _, err := Load(nil); err == nil {
		t.Fatalf("Load(nil) error = %v, want validation error", err)
	}
}

func TestSaveLoadRoundTripL2(t *testing.T) {
	cfg := Config{
		Dim:               3,
		M:                 6,
		K0:                12,
		CandidateK:        12,
		ConstructionL:     18,
		Alpha:             67,
		Iterations:        2,
		Seed:              17,
		Workers:           1,
		CandidateRecall:   0.90,
		CandidateControls: 12,
	}
	idx := mustBuildIndex(t, cfg, synth.UniformVectors(40, 3))
	query := []float32{1.25, 2.5, 3.75}

	before, err := idx.Search(query, 5, 18)
	if err != nil {
		t.Fatalf("Search before save returned error: %v", err)
	}
	loaded := roundTripIndex(t, idx)
	assertIndexState(t, loaded, idx)
	after, err := loaded.Search(query, 5, 18)
	if err != nil {
		t.Fatalf("Search after load returned error: %v", err)
	}
	assertResults(t, after, before)
}

func TestSaveLoadRoundTripCosine(t *testing.T) {
	cfg := Config{
		Metric:            MetricCosine,
		Dim:               2,
		M:                 4,
		K0:                8,
		CandidateK:        8,
		ConstructionL:     12,
		Alpha:             67,
		Iterations:        2,
		Seed:              29,
		Workers:           1,
		CandidateRecall:   0.90,
		CandidateControls: 8,
	}
	idx := mustBuildIndex(t, cfg, [][]float32{
		{3, 4},
		{5, 12},
		{8, 15},
		{7, 24},
		{9, 40},
		{20, 21},
	})
	query := []float32{6, 8}

	before, err := idx.Search(query, 3, 8)
	if err != nil {
		t.Fatalf("Search before save returned error: %v", err)
	}
	loaded := roundTripIndex(t, idx)
	assertIndexState(t, loaded, idx)
	for id := 0; id < loaded.count; id++ {
		vector := vectorAt(loaded.vectors, loaded.dim, id)
		var norm float64
		for _, value := range vector {
			norm += float64(value * value)
		}
		if math.Abs(norm-1) > 1e-5 {
			t.Fatalf("loaded cosine vector %d norm = %v, want 1", id, norm)
		}
	}
	after, err := loaded.Search(query, 3, 8)
	if err != nil {
		t.Fatalf("Search after load returned error: %v", err)
	}
	assertResults(t, after, before)
}

func TestLoadRejectsCorruptedInput(t *testing.T) {
	valid := writeRawPersistenceIndex(t, validRawPersistenceIndex())

	tests := []struct {
		name string
		data []byte
	}{
		{name: "bad magic", data: []byte("not an index")},
		{name: "unsupported version", data: withUint32(valid, len(persistenceMagic), 99)},
		{name: "truncated input", data: valid[:len(valid)-3]},
		{name: "invalid metric", data: withUint32(valid, len(persistenceMagic)+4, 99)},
		{name: "invalid dimension", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.dim = 0
		})},
		{name: "invalid vector count", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.count = 0
			raw.vectors = nil
			raw.levels = nil
			raw.layers = [][][]int{{}}
		})},
		{name: "invalid alpha", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.cfg.Alpha = math.NaN()
		})},
		{name: "missing alpha", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.cfg.Alpha = 0
		})},
		{name: "missing seed", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.cfg.Seed = 0
		})},
		{name: "missing candidate recall", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.cfg.CandidateRecall = 0
		})},
		{name: "invalid entry point", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.entryPoint = raw.count
		})},
		{name: "invalid neighbor id", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.layers[0][0] = []int{raw.count}
		})},
		{name: "duplicate neighbor", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.layers[0][0] = []int{1, 1}
		})},
		{name: "self neighbor", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.layers[0][0] = []int{0}
		})},
		{name: "degree overflow", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.count = 4
			raw.vectors = []float32{0, 0, 1, 0, 0, 1, 1, 1}
			raw.levels = []int{1, 1, 1, 1}
			raw.entryPoint = 0
			raw.maxLayer = 1
			raw.layers = [][][]int{
				{{1}, {0}, {0}, {0}},
				{{1, 2, 3}, {0}, {0}, {0}},
			}
		})},
		{name: "source below layer has adjacency", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.layers[1][1] = []int{0}
		})},
		{name: "neighbor below layer", data: rawPersistenceBytes(t, func(raw *rawPersistenceIndex) {
			raw.layers[1][0] = []int{1}
		})},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := Load(bytes.NewReader(tt.data)); err == nil {
				t.Fatal("Load returned nil error")
			}
		})
	}
}

func roundTripIndex(t *testing.T, idx *Index) *Index {
	t.Helper()

	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	loaded, err := Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	return loaded
}

func assertIndexState(t *testing.T, got, want *Index) {
	t.Helper()

	if got.cfg != want.cfg {
		t.Fatalf("loaded config = %+v, want %+v", got.cfg, want.cfg)
	}
	if got.dim != want.dim || got.count != want.count || got.entryPoint != want.entryPoint || got.maxLayer != want.maxLayer {
		t.Fatalf("loaded metadata = dim:%d count:%d entry:%d maxLayer:%d, want dim:%d count:%d entry:%d maxLayer:%d",
			got.dim, got.count, got.entryPoint, got.maxLayer,
			want.dim, want.count, want.entryPoint, want.maxLayer)
	}
	if !got.graphReady {
		t.Fatal("loaded graphReady = false, want true")
	}
	if !reflect.DeepEqual(got.vectors, want.vectors) {
		t.Fatalf("loaded vectors differ")
	}
	if !reflect.DeepEqual(got.levels, want.levels) {
		t.Fatalf("loaded levels = %v, want %v", got.levels, want.levels)
	}
	if !reflect.DeepEqual(got.layers, want.layers) {
		t.Fatalf("loaded layers differ")
	}
}

type rawPersistenceIndex struct {
	metric     Metric
	dim        int
	cfg        Config
	count      int
	vectors    []float32
	levels     []int
	entryPoint int
	maxLayer   int
	layers     [][][]int
}

func validRawPersistenceIndex() rawPersistenceIndex {
	cfg, err := normalizeConfig(Config{
		Dim:               2,
		M:                 2,
		K0:                2,
		CandidateK:        2,
		ConstructionL:     3,
		Alpha:             67,
		Iterations:        1,
		Seed:              5,
		Workers:           1,
		CandidateRecall:   0.90,
		CandidateControls: 2,
	})
	if err != nil {
		panic(err)
	}
	return rawPersistenceIndex{
		metric:     MetricL2,
		dim:        2,
		cfg:        cfg,
		count:      3,
		vectors:    []float32{0, 0, 1, 0, 0, 1},
		levels:     []int{1, 0, 0},
		entryPoint: 0,
		maxLayer:   1,
		layers: [][][]int{
			{{1, 2}, {0, 2}, {0, 1}},
			{{}, {}, {}},
		},
	}
}

func rawPersistenceBytes(t *testing.T, mutate func(*rawPersistenceIndex)) []byte {
	t.Helper()

	raw := validRawPersistenceIndex()
	mutate(&raw)
	return writeRawPersistenceIndex(t, raw)
}

func writeRawPersistenceIndex(t *testing.T, raw rawPersistenceIndex) []byte {
	t.Helper()

	var buf bytes.Buffer
	mustWritePersistence(t, writeFull(&buf, persistenceMagic[:]))
	mustWritePersistence(t, writeUint32(&buf, persistenceVersion))
	mustWritePersistence(t, writeUint32(&buf, uint32(raw.metric)))
	mustWritePersistence(t, writeUint32(&buf, uint32(raw.dim)))
	mustWritePersistence(t, writeConfig(&buf, raw.cfg))
	mustWritePersistence(t, writeUint32(&buf, uint32(raw.count)))
	for _, value := range raw.vectors {
		mustWritePersistence(t, writeUint32(&buf, math.Float32bits(value)))
	}
	for _, level := range raw.levels {
		mustWritePersistence(t, writeUint32(&buf, uint32(level)))
	}
	mustWritePersistence(t, writeInt32(&buf, int32(raw.entryPoint)))
	mustWritePersistence(t, writeInt32(&buf, int32(raw.maxLayer)))
	mustWritePersistence(t, writeUint32(&buf, uint32(len(raw.layers))))
	for layer := range raw.layers {
		for sourceID := 0; sourceID < raw.count && sourceID < len(raw.layers[layer]); sourceID++ {
			neighbors := raw.layers[layer][sourceID]
			mustWritePersistence(t, writeUint32(&buf, uint32(len(neighbors))))
			for _, neighborID := range neighbors {
				mustWritePersistence(t, writeUint32(&buf, uint32(neighborID)))
			}
		}
	}
	return buf.Bytes()
}

func mustWritePersistence(t *testing.T, err error) {
	t.Helper()

	if err != nil {
		t.Fatalf("write persistence test data: %v", err)
	}
}

func withUint32(data []byte, offset int, value uint32) []byte {
	out := append([]byte(nil), data...)
	persistenceOrder.PutUint32(out[offset:], value)
	return out
}
