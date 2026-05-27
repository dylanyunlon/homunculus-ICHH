package benchdata

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/cryo-zd/fasthnsw"
	"github.com/scigolib/hdf5"
)

var hdf5DimPattern = regexp.MustCompile(`2D array \[(\d+) x (\d+)\]`)

// LoadANNBenchHDF5 loads ANN-Benchmarks-style HDF5 files. The expected
// datasets are train, test, and neighbors.
func LoadANNBenchHDF5(path string, metric fasthnsw.Metric, limitBase int, limitQueries int) (Dataset, error) {
	file, err := hdf5.Open(path)
	if err != nil {
		return Dataset{}, fmt.Errorf("open hdf5 file: %w", err)
	}
	defer file.Close()

	train, err := findHDF5Dataset(file, "/train")
	if err != nil {
		return Dataset{}, err
	}
	test, err := findHDF5Dataset(file, "/test")
	if err != nil {
		return Dataset{}, err
	}
	neighbors, err := findHDF5Dataset(file, "/neighbors")
	if err != nil {
		return Dataset{}, err
	}

	base, err := readHDF5FloatMatrix(train, limitBase)
	if err != nil {
		return Dataset{}, fmt.Errorf("read train: %w", err)
	}
	queries, err := readHDF5FloatMatrix(test, limitQueries)
	if err != nil {
		return Dataset{}, fmt.Errorf("read test: %w", err)
	}
	truth, err := readHDF5IntMatrix(neighbors, limitQueries)
	if err != nil {
		return Dataset{}, fmt.Errorf("read neighbors: %w", err)
	}
	return Dataset{
		Name:        "ann-benchmarks-hdf5",
		Metric:      metric,
		Base:        base,
		Queries:     queries,
		GroundTruth: truth,
	}, nil
}

func findHDF5Dataset(file *hdf5.File, path string) (*hdf5.Dataset, error) {
	var found *hdf5.Dataset
	file.Walk(func(currentPath string, obj hdf5.Object) {
		if found != nil {
			return
		}
		if currentPath == path || strings.TrimPrefix(currentPath, "/") == strings.TrimPrefix(path, "/") {
			if dataset, ok := obj.(*hdf5.Dataset); ok {
				found = dataset
			}
		}
	})
	if found == nil {
		return nil, fmt.Errorf("hdf5 dataset %s not found", path)
	}
	return found, nil
}

func readHDF5FloatMatrix(dataset *hdf5.Dataset, limitRows int) ([][]float32, error) {
	rows, cols, err := hdf5MatrixDimensions(dataset)
	if err != nil {
		return nil, err
	}
	if limitRows > 0 && uint64(limitRows) < rows {
		rows = uint64(limitRows)
	}
	raw, err := dataset.ReadSlice([]uint64{0, 0}, []uint64{rows, cols})
	if err != nil {
		return nil, err
	}
	values, err := hdf5FloatValues(raw)
	if err != nil {
		return nil, err
	}
	return reshapeFloat32(values, int(rows), int(cols))
}

func readHDF5IntMatrix(dataset *hdf5.Dataset, limitRows int) ([][]int, error) {
	rows, cols, err := hdf5MatrixDimensions(dataset)
	if err != nil {
		return nil, err
	}
	if limitRows > 0 && uint64(limitRows) < rows {
		rows = uint64(limitRows)
	}
	raw, err := dataset.ReadSlice([]uint64{0, 0}, []uint64{rows, cols})
	if err != nil {
		return nil, err
	}
	values, err := hdf5IntValues(raw)
	if err != nil {
		return nil, err
	}
	return reshapeInts(values, int(rows), int(cols))
}

func hdf5MatrixDimensions(dataset *hdf5.Dataset) (uint64, uint64, error) {
	info, err := dataset.Info()
	if err != nil {
		return 0, 0, err
	}
	match := hdf5DimPattern.FindStringSubmatch(info)
	if len(match) != 3 {
		return 0, 0, fmt.Errorf("dataset %s is not a 2D matrix: %s", dataset.Name(), info)
	}
	rows, err := strconv.ParseUint(match[1], 10, 64)
	if err != nil {
		return 0, 0, err
	}
	cols, err := strconv.ParseUint(match[2], 10, 64)
	if err != nil {
		return 0, 0, err
	}
	if rows == 0 || cols == 0 {
		return 0, 0, fmt.Errorf("dataset %s has invalid dimensions %dx%d", dataset.Name(), rows, cols)
	}
	return rows, cols, nil
}

func hdf5FloatValues(raw interface{}) ([]float32, error) {
	switch values := raw.(type) {
	case []float32:
		out := make([]float32, len(values))
		copy(out, values)
		return out, nil
	case []float64:
		out := make([]float32, len(values))
		for i, value := range values {
			out[i] = float32(value)
		}
		return out, nil
	case []int32:
		out := make([]float32, len(values))
		for i, value := range values {
			out[i] = float32(value)
		}
		return out, nil
	case []int64:
		out := make([]float32, len(values))
		for i, value := range values {
			out[i] = float32(value)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported hdf5 float matrix type %T", raw)
	}
}

func hdf5IntValues(raw interface{}) ([]int, error) {
	switch values := raw.(type) {
	case []int:
		out := make([]int, len(values))
		copy(out, values)
		return out, nil
	case []int32:
		out := make([]int, len(values))
		for i, value := range values {
			if value < 0 {
				return nil, fmt.Errorf("negative neighbor id %d", value)
			}
			out[i] = int(value)
		}
		return out, nil
	case []int64:
		out := make([]int, len(values))
		for i, value := range values {
			if value < 0 {
				return nil, fmt.Errorf("negative neighbor id %d", value)
			}
			out[i] = int(value)
		}
		return out, nil
	case []float64:
		out := make([]int, len(values))
		for i, value := range values {
			if value < 0 || value != float64(int(value)) {
				return nil, fmt.Errorf("invalid neighbor id %v", value)
			}
			out[i] = int(value)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported hdf5 integer matrix type %T", raw)
	}
}

func reshapeFloat32(values []float32, rows int, cols int) ([][]float32, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("matrix dimensions must be positive")
	}
	if len(values) != rows*cols {
		return nil, fmt.Errorf("matrix value count %d does not match dimensions %dx%d", len(values), rows, cols)
	}
	out := make([][]float32, rows)
	for row := 0; row < rows; row++ {
		vector := make([]float32, cols)
		copy(vector, values[row*cols:(row+1)*cols])
		out[row] = vector
	}
	return out, nil
}

func reshapeInts(values []int, rows int, cols int) ([][]int, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("matrix dimensions must be positive")
	}
	if len(values) != rows*cols {
		return nil, fmt.Errorf("matrix value count %d does not match dimensions %dx%d", len(values), rows, cols)
	}
	out := make([][]int, rows)
	for row := 0; row < rows; row++ {
		vector := make([]int, cols)
		copy(vector, values[row*cols:(row+1)*cols])
		out[row] = vector
	}
	return out, nil
}
