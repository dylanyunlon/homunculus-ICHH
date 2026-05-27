package benchdata

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/cryo-zd/fasthnsw"
	"github.com/scigolib/hdf5"
)

func TestLoadANNBenchHDF5(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ann.hdf5")
	writeANNBenchHDF5Fixture(t, path)

	got, err := LoadANNBenchHDF5(path, fasthnsw.MetricL2, 2, 1)
	if err != nil {
		t.Fatalf("LoadANNBenchHDF5 returned error: %v", err)
	}
	if got.Name != "ann-benchmarks-hdf5" || got.Dim() != 2 {
		t.Fatalf("dataset metadata = name:%s dim:%d", got.Name, got.Dim())
	}
	if len(got.Base) != 2 || len(got.Queries) != 1 {
		t.Fatalf("dataset sizes = base:%d queries:%d", len(got.Base), len(got.Queries))
	}
	if !reflect.DeepEqual(got.GroundTruth, [][]int{{0, 1}}) {
		t.Fatalf("ground truth = %v", got.GroundTruth)
	}
}

func writeANNBenchHDF5Fixture(t *testing.T, path string) {
	t.Helper()

	file, err := hdf5.CreateForWrite(path, hdf5.CreateTruncate)
	if err != nil {
		t.Fatalf("CreateForWrite returned error: %v", err)
	}

	train, err := file.CreateDataset("/train", hdf5.Float32, []uint64{3, 2})
	if err != nil {
		t.Fatalf("CreateDataset train returned error: %v", err)
	}
	if err := train.Write([]float32{0, 0, 1, 0, 0, 1}); err != nil {
		t.Fatalf("write train: %v", err)
	}

	test, err := file.CreateDataset("/test", hdf5.Float32, []uint64{2, 2})
	if err != nil {
		t.Fatalf("CreateDataset test returned error: %v", err)
	}
	if err := test.Write([]float32{0, 0, 1, 1}); err != nil {
		t.Fatalf("write test: %v", err)
	}

	neighbors, err := file.CreateDataset("/neighbors", hdf5.Int32, []uint64{2, 2})
	if err != nil {
		t.Fatalf("CreateDataset neighbors returned error: %v", err)
	}
	if err := neighbors.Write([]int32{0, 1, 1, 2}); err != nil {
		t.Fatalf("write neighbors: %v", err)
	}

	if err := file.Close(); err != nil {
		t.Fatalf("close fixture: %v", err)
	}
}
