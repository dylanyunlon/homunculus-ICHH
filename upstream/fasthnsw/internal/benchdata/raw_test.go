package benchdata

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cryo-zd/fasthnsw"
)

func TestLoadRawDatasetFVECS(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.fvecs")
	queries := filepath.Join(dir, "query.fvecs")
	truth := filepath.Join(dir, "truth.ivecs")
	writeFVECSFile(t, base, [][]float32{{0, 0}, {1, 0}, {0, 1}})
	writeFVECSFile(t, queries, [][]float32{{0, 0}})
	writeIVECSFile(t, truth, [][]int{{0, 1}})

	got, err := LoadRawDataset("fvecs", base, queries, truth, fasthnsw.MetricL2, 0, 0)
	if err != nil {
		t.Fatalf("LoadRawDataset returned error: %v", err)
	}
	if got.Name != "fvecs" || got.Dim() != 2 {
		t.Fatalf("dataset metadata = name:%s dim:%d", got.Name, got.Dim())
	}
	if !reflect.DeepEqual(got.GroundTruth, [][]int{{0, 1}}) {
		t.Fatalf("ground truth = %v", got.GroundTruth)
	}
}

func TestLoadBVECS(t *testing.T) {
	path := filepath.Join(t.TempDir(), "base.bvecs")
	writeBVECSFile(t, path, [][]byte{{1, 2}, {3, 4}})

	got, err := LoadBVECS(path, 0)
	if err != nil {
		t.Fatalf("LoadBVECS returned error: %v", err)
	}
	want := [][]float32{{1, 2}, {3, 4}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("LoadBVECS = %v, want %v", got, want)
	}
}

func TestRawLoadersRejectBadInput(t *testing.T) {
	t.Run("truncated fvecs", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "bad.fvecs")
		var buf bytes.Buffer
		mustWrite(t, binary.Write(&buf, binary.LittleEndian, int32(2)))
		mustWrite(t, binary.Write(&buf, binary.LittleEndian, float32(1)))
		mustWriteFile(t, path, buf.Bytes())

		if _, err := LoadFVECS(path, 0); err == nil {
			t.Fatal("LoadFVECS returned nil error")
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "bad.fvecs")
		var buf bytes.Buffer
		writeFVECSRecord(t, &buf, []float32{1, 2})
		writeFVECSRecord(t, &buf, []float32{1})
		mustWriteFile(t, path, buf.Bytes())

		_, err := LoadFVECS(path, 0)
		if err == nil || !strings.Contains(err.Error(), "dimension") {
			t.Fatalf("LoadFVECS error = %v, want dimension error", err)
		}
	})

	t.Run("invalid dimension", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "bad.fvecs")
		var buf bytes.Buffer
		mustWrite(t, binary.Write(&buf, binary.LittleEndian, int32(0)))
		mustWriteFile(t, path, buf.Bytes())

		_, err := LoadFVECS(path, 0)
		if err == nil || !strings.Contains(err.Error(), "dimension") {
			t.Fatalf("LoadFVECS error = %v, want dimension error", err)
		}
	})
}

func TestLoadSyntheticDataset(t *testing.T) {
	got, err := LoadSynthetic(SyntheticOptions{
		Shape:    "clustered",
		Vectors:  24,
		Queries:  6,
		Dim:      3,
		Clusters: 3,
		Metric:   fasthnsw.MetricL2,
		K:        2,
	})
	if err != nil {
		t.Fatalf("LoadSynthetic returned error: %v", err)
	}
	if err := got.Validate(2); err != nil {
		t.Fatalf("synthetic dataset did not validate: %v", err)
	}
	if got.GroundTruth[0][0] < 0 {
		t.Fatalf("invalid ground truth: %v", got.GroundTruth)
	}
}

func TestExactGroundTruthL2TieBreaksByID(t *testing.T) {
	got, err := ExactGroundTruth(
		[][]float32{{1, 0}, {-1, 0}, {0, 1}},
		[][]float32{{0, 0}},
		fasthnsw.MetricL2,
		3,
	)
	if err != nil {
		t.Fatalf("ExactGroundTruth returned error: %v", err)
	}
	want := [][]int{{0, 1, 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ExactGroundTruth = %v, want %v", got, want)
	}
}

func TestExactGroundTruthCosineNormalizesThroughCore(t *testing.T) {
	got, err := ExactGroundTruth(
		[][]float32{{10, 0}, {0, 3}, {-2, 0}},
		[][]float32{{5, 0}},
		fasthnsw.MetricCosine,
		2,
	)
	if err != nil {
		t.Fatalf("ExactGroundTruth returned error: %v", err)
	}
	want := [][]int{{0, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ExactGroundTruth = %v, want %v", got, want)
	}
}

func TestExactGroundTruthRejectsCosineZeroVector(t *testing.T) {
	_, err := ExactGroundTruth(
		[][]float32{{1, 0}, {0, 1}},
		[][]float32{{0, 0}},
		fasthnsw.MetricCosine,
		1,
	)
	if err == nil || !strings.Contains(err.Error(), "query 0") {
		t.Fatalf("ExactGroundTruth error = %v, want query zero-vector error", err)
	}
}

func writeFVECSFile(t *testing.T, path string, vectors [][]float32) {
	t.Helper()
	var buf bytes.Buffer
	for _, vector := range vectors {
		writeFVECSRecord(t, &buf, vector)
	}
	mustWriteFile(t, path, buf.Bytes())
}

func writeFVECSRecord(t *testing.T, buf *bytes.Buffer, vector []float32) {
	t.Helper()
	mustWrite(t, binary.Write(buf, binary.LittleEndian, int32(len(vector))))
	for _, value := range vector {
		mustWrite(t, binary.Write(buf, binary.LittleEndian, value))
	}
}

func writeBVECSFile(t *testing.T, path string, vectors [][]byte) {
	t.Helper()
	var buf bytes.Buffer
	for _, vector := range vectors {
		mustWrite(t, binary.Write(&buf, binary.LittleEndian, int32(len(vector))))
		_, err := buf.Write(vector)
		mustWrite(t, err)
	}
	mustWriteFile(t, path, buf.Bytes())
}

func writeIVECSFile(t *testing.T, path string, rows [][]int) {
	t.Helper()
	var buf bytes.Buffer
	for _, row := range rows {
		mustWrite(t, binary.Write(&buf, binary.LittleEndian, int32(len(row))))
		for _, value := range row {
			mustWrite(t, binary.Write(&buf, binary.LittleEndian, int32(value)))
		}
	}
	mustWriteFile(t, path, buf.Bytes())
}

func mustWriteFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}
}

func mustWrite(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}
