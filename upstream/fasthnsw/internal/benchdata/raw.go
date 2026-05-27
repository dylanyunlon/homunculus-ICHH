package benchdata

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/cryo-zd/fasthnsw"
)

const maxRawDimension = 1 << 20

// LoadRawDataset loads SIFT-style raw vector files and an ivecs ground-truth
// file. kind must be fvecs or bvecs.
func LoadRawDataset(kind string, basePath string, queriesPath string, truthPath string, metric fasthnsw.Metric, limitBase int, limitQueries int) (Dataset, error) {
	if basePath == "" || queriesPath == "" || truthPath == "" {
		return Dataset{}, fmt.Errorf("%s validation requires -base, -queries, and -truth", kind)
	}

	var base [][]float32
	var queries [][]float32
	var err error
	switch kind {
	case "fvecs":
		base, err = LoadFVECS(basePath, limitBase)
		if err != nil {
			return Dataset{}, err
		}
		queries, err = LoadFVECS(queriesPath, limitQueries)
		if err != nil {
			return Dataset{}, err
		}
	case "bvecs":
		base, err = LoadBVECS(basePath, limitBase)
		if err != nil {
			return Dataset{}, err
		}
		queries, err = LoadBVECS(queriesPath, limitQueries)
		if err != nil {
			return Dataset{}, err
		}
	default:
		return Dataset{}, fmt.Errorf("unsupported raw dataset kind %q", kind)
	}

	truth, err := LoadIVECS(truthPath, limitQueries)
	if err != nil {
		return Dataset{}, err
	}
	return Dataset{
		Name:        kind,
		Metric:      metric,
		Base:        base,
		Queries:     queries,
		GroundTruth: truth,
	}, nil
}

// LoadFVECS reads little-endian .fvecs records.
func LoadFVECS(path string, limit int) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open fvecs file: %w", err)
	}
	defer file.Close()
	vectors, err := readRawVectors(file, limit, readFVECSRecord)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return vectors, nil
}

// LoadBVECS reads little-endian .bvecs records and converts bytes to float32.
func LoadBVECS(path string, limit int) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open bvecs file: %w", err)
	}
	defer file.Close()
	vectors, err := readRawVectors(file, limit, readBVECSRecord)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return vectors, nil
}

// LoadIVECS reads little-endian .ivecs records.
func LoadIVECS(path string, limit int) ([][]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open ivecs file: %w", err)
	}
	defer file.Close()
	vectors, err := readIVECS(file, limit)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return vectors, nil
}

type rawVectorReader func(io.Reader, int) ([]float32, error)

func readRawVectors(r io.Reader, limit int, readRecord rawVectorReader) ([][]float32, error) {
	if limit < 0 {
		return nil, fmt.Errorf("limit must be non-negative")
	}
	var vectors [][]float32
	dim := 0
	for limit == 0 || len(vectors) < limit {
		recordDim, err := readRawDim(r)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if dim == 0 {
			dim = recordDim
		}
		if recordDim != dim {
			return nil, fmt.Errorf("record %d has dimension %d, want %d", len(vectors), recordDim, dim)
		}
		vector, err := readRecord(r, dim)
		if err != nil {
			return nil, fmt.Errorf("record %d: %w", len(vectors), err)
		}
		vectors = append(vectors, vector)
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("raw vector file has no vectors")
	}
	return vectors, nil
}

func readIVECS(r io.Reader, limit int) ([][]int, error) {
	if limit < 0 {
		return nil, fmt.Errorf("limit must be non-negative")
	}
	var rows [][]int
	dim := 0
	for limit == 0 || len(rows) < limit {
		recordDim, err := readRawDim(r)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if dim == 0 {
			dim = recordDim
		}
		if recordDim != dim {
			return nil, fmt.Errorf("record %d has dimension %d, want %d", len(rows), recordDim, dim)
		}
		values := make([]int32, dim)
		if err := binary.Read(r, binary.LittleEndian, values); err != nil {
			return nil, fmt.Errorf("record %d: %w", len(rows), err)
		}
		row := make([]int, dim)
		for i, value := range values {
			if value < 0 {
				return nil, fmt.Errorf("record %d contains negative neighbor id %d", len(rows), value)
			}
			row[i] = int(value)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("ivecs file has no rows")
	}
	return rows, nil
}

func readRawDim(r io.Reader) (int, error) {
	var dim int32
	err := binary.Read(r, binary.LittleEndian, &dim)
	if err != nil {
		return 0, err
	}
	if dim <= 0 {
		return 0, fmt.Errorf("record dimension must be positive")
	}
	if dim > maxRawDimension {
		return 0, fmt.Errorf("record dimension %d exceeds limit %d", dim, maxRawDimension)
	}
	return int(dim), nil
}

func readFVECSRecord(r io.Reader, dim int) ([]float32, error) {
	vector := make([]float32, dim)
	if err := binary.Read(r, binary.LittleEndian, vector); err != nil {
		return nil, err
	}
	return vector, nil
}

func readBVECSRecord(r io.Reader, dim int) ([]float32, error) {
	values := make([]byte, dim)
	if _, err := io.ReadFull(r, values); err != nil {
		return nil, err
	}
	vector := make([]float32, dim)
	for i, value := range values {
		vector[i] = float32(value)
	}
	return vector, nil
}
