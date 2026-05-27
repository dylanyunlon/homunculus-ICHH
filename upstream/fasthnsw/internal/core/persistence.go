package core

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const (
	persistenceVersion = uint32(1)

	maxPersistedCount        = 1<<31 - 1
	maxPersistedDimension    = 1 << 20
	maxPersistedVectorValues = 1 << 30
	maxPersistedLayers       = 1024
	maxPersistedDegree       = 1 << 20
)

var (
	persistenceMagic = [...]byte{'F', 'H', 'N', 'S', 'W', 'I', 'D', 'X'}
	persistenceOrder = binary.LittleEndian
)

// save writes the index-owned representation directly. It intentionally does
// not rebuild, re-normalize, or re-prune data: persistence should preserve the
// exact graph that Search used before saving.
func (idx *Index) save(w io.Writer) error {
	if err := validatePersistableIndex(idx); err != nil {
		return err
	}

	if err := writeFull(w, persistenceMagic[:]); err != nil {
		return fmt.Errorf("fasthnsw: write persistence magic: %w", err)
	}
	if err := writeUint32(w, persistenceVersion); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(idx.cfg.Metric)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(idx.dim)); err != nil {
		return err
	}
	if err := writeConfig(w, idx.cfg); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(idx.count)); err != nil {
		return err
	}
	for _, value := range idx.vectors {
		if err := writeUint32(w, math.Float32bits(value)); err != nil {
			return err
		}
	}
	for _, level := range idx.levels {
		if err := writeUint32(w, uint32(level)); err != nil {
			return err
		}
	}
	if err := writeInt32(w, int32(idx.entryPoint)); err != nil {
		return err
	}
	if err := writeInt32(w, int32(idx.maxLayer)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(len(idx.layers))); err != nil {
		return err
	}
	for layer := range idx.layers {
		for sourceID := range idx.layers[layer] {
			neighbors := idx.layers[layer][sourceID]
			if err := writeUint32(w, uint32(len(neighbors))); err != nil {
				return err
			}
			for _, neighborID := range neighbors {
				if err := writeUint32(w, uint32(neighborID)); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// loadIndex reads a versioned binary index and validates the graph before it
// becomes searchable. Count and degree checks happen before large slices are
// allocated so corrupted input returns an error instead of driving unbounded
// allocation.
func loadIndex(r io.Reader) (*Index, error) {
	var magic [len(persistenceMagic)]byte
	if err := readFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("fasthnsw: read persistence magic: %w", err)
	}
	if magic != persistenceMagic {
		return nil, fmt.Errorf("fasthnsw: invalid persistence magic")
	}

	version, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("fasthnsw: read persistence version: %w", err)
	}
	if version != persistenceVersion {
		return nil, fmt.Errorf("fasthnsw: unsupported persistence version %d", version)
	}

	metricValue, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("fasthnsw: read metric: %w", err)
	}
	metric := Metric(metricValue)
	if !validMetric(metric) {
		return nil, fmt.Errorf("fasthnsw: unsupported metric %d", metricValue)
	}

	dimValue, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("fasthnsw: read dimension: %w", err)
	}
	dim, err := persistedPositiveInt(dimValue, "dimension", maxPersistedDimension)
	if err != nil {
		return nil, err
	}

	cfg, err := readConfig(r, metric, dim)
	if err != nil {
		return nil, err
	}

	countValue, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("fasthnsw: read vector count: %w", err)
	}
	count, err := persistedPositiveInt(countValue, "vector count", maxPersistedCount)
	if err != nil {
		return nil, err
	}
	vectorValues, err := checkedVectorValueCount(count, dim)
	if err != nil {
		return nil, err
	}

	vectors := make([]float32, vectorValues)
	for i := range vectors {
		bits, err := readUint32(r)
		if err != nil {
			return nil, fmt.Errorf("fasthnsw: read vector value %d: %w", i, err)
		}
		vectors[i] = math.Float32frombits(bits)
	}

	levels := make([]int, count)
	maxSeenLevel := 0
	for i := range levels {
		levelValue, err := readUint32(r)
		if err != nil {
			return nil, fmt.Errorf("fasthnsw: read level %d: %w", i, err)
		}
		level, err := persistedNonNegativeInt(levelValue, "level", maxPersistedLayers-1)
		if err != nil {
			return nil, err
		}
		levels[i] = level
		if level > maxSeenLevel {
			maxSeenLevel = level
		}
	}

	entryPoint, err := readInt32(r)
	if err != nil {
		return nil, fmt.Errorf("fasthnsw: read entry point: %w", err)
	}
	maxLayer, err := readInt32(r)
	if err != nil {
		return nil, fmt.Errorf("fasthnsw: read max layer: %w", err)
	}
	if maxLayer < 0 {
		return nil, fmt.Errorf("fasthnsw: max layer must be non-negative")
	}
	if int(maxLayer) >= maxPersistedLayers {
		return nil, fmt.Errorf("fasthnsw: max layer %d exceeds persistence limit %d", maxLayer, maxPersistedLayers-1)
	}

	layerCountValue, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("fasthnsw: read layer count: %w", err)
	}
	layerCount, err := persistedPositiveInt(layerCountValue, "layer count", maxPersistedLayers)
	if err != nil {
		return nil, err
	}
	if layerCount != int(maxLayer)+1 {
		return nil, fmt.Errorf("fasthnsw: layer count %d does not match max layer %d", layerCount, maxLayer)
	}

	layers := make([][][]int, layerCount)
	for layer := range layers {
		layers[layer] = make([][]int, count)
		for sourceID := 0; sourceID < count; sourceID++ {
			degreeValue, err := readUint32(r)
			if err != nil {
				return nil, fmt.Errorf("fasthnsw: read layer %d node %d degree: %w", layer, sourceID, err)
			}
			degree, err := persistedNonNegativeInt(degreeValue, "neighbor count", maxPersistedDegree)
			if err != nil {
				return nil, err
			}
			if degree > layerMaxDegree(cfg, layer) {
				return nil, fmt.Errorf("fasthnsw: layer %d node %d has degree %d above bound %d", layer, sourceID, degree, layerMaxDegree(cfg, layer))
			}
			var neighbors []int
			if degree > 0 {
				neighbors = make([]int, degree)
			}
			seen := make(map[int]bool, degree)
			for i := range neighbors {
				neighborValue, err := readUint32(r)
				if err != nil {
					return nil, fmt.Errorf("fasthnsw: read layer %d node %d neighbor %d: %w", layer, sourceID, i, err)
				}
				neighborID, err := persistedNonNegativeInt(neighborValue, "neighbor id", count-1)
				if err != nil {
					return nil, err
				}
				if neighborID == sourceID {
					return nil, fmt.Errorf("fasthnsw: layer %d node %d contains self-neighbor", layer, sourceID)
				}
				if seen[neighborID] {
					return nil, fmt.Errorf("fasthnsw: layer %d node %d contains duplicate neighbor %d", layer, sourceID, neighborID)
				}
				seen[neighborID] = true
				neighbors[i] = neighborID
			}
			layers[layer][sourceID] = neighbors
		}
	}

	idx := &Index{
		cfg:        cfg,
		vectors:    vectors,
		dim:        dim,
		count:      count,
		layers:     layers,
		levels:     levels,
		entryPoint: int(entryPoint),
		maxLayer:   int(maxLayer),
		graphReady: true,
	}
	if err := validateLoadedIndex(idx, maxSeenLevel); err != nil {
		return nil, err
	}
	return idx, nil
}

func writeConfig(w io.Writer, cfg Config) error {
	values := []uint32{
		uint32(cfg.M),
		uint32(cfg.EfConstruction),
		uint32(cfg.K0),
		uint32(cfg.CandidateK),
		uint32(cfg.ConstructionL),
	}
	for _, value := range values {
		if err := writeUint32(w, value); err != nil {
			return err
		}
	}
	if err := writeUint64(w, math.Float64bits(cfg.Alpha)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(cfg.Iterations)); err != nil {
		return err
	}
	if err := writeInt64(w, cfg.Seed); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(cfg.Workers)); err != nil {
		return err
	}
	if err := writeUint64(w, math.Float64bits(cfg.CandidateRecall)); err != nil {
		return err
	}
	return writeUint32(w, uint32(cfg.CandidateControls))
}

func readConfig(r io.Reader, metric Metric, dim int) (Config, error) {
	m, err := readRequiredInt(r, "M", maxPersistedDegree)
	if err != nil {
		return Config{}, err
	}
	efConstruction, err := readRequiredInt(r, "EfConstruction", maxPersistedDegree)
	if err != nil {
		return Config{}, err
	}
	k0, err := readRequiredInt(r, "K0", maxPersistedDegree)
	if err != nil {
		return Config{}, err
	}
	candidateK, err := readRequiredInt(r, "CandidateK", maxPersistedDegree)
	if err != nil {
		return Config{}, err
	}
	constructionL, err := readRequiredInt(r, "ConstructionL", maxPersistedDegree)
	if err != nil {
		return Config{}, err
	}
	alphaBits, err := readUint64(r)
	if err != nil {
		return Config{}, fmt.Errorf("fasthnsw: read Alpha: %w", err)
	}
	alpha := math.Float64frombits(alphaBits)
	if alpha == 0 {
		return Config{}, fmt.Errorf("fasthnsw: Alpha must be present in persisted config")
	}
	iterations, err := readRequiredInt(r, "Iterations", maxPersistedDegree)
	if err != nil {
		return Config{}, err
	}
	seed, err := readInt64(r)
	if err != nil {
		return Config{}, fmt.Errorf("fasthnsw: read Seed: %w", err)
	}
	if seed == 0 {
		return Config{}, fmt.Errorf("fasthnsw: Seed must be present in persisted config")
	}
	workers, err := readRequiredInt(r, "Workers", maxPersistedDegree)
	if err != nil {
		return Config{}, err
	}
	candidateRecallBits, err := readUint64(r)
	if err != nil {
		return Config{}, fmt.Errorf("fasthnsw: read CandidateRecall: %w", err)
	}
	candidateRecall := math.Float64frombits(candidateRecallBits)
	if candidateRecall == 0 {
		return Config{}, fmt.Errorf("fasthnsw: CandidateRecall must be present in persisted config")
	}
	candidateControls, err := readRequiredInt(r, "CandidateControls", maxPersistedDegree)
	if err != nil {
		return Config{}, err
	}

	cfg, err := normalizeConfig(Config{
		Metric:            metric,
		Dim:               dim,
		M:                 m,
		EfConstruction:    efConstruction,
		K0:                k0,
		CandidateK:        candidateK,
		ConstructionL:     constructionL,
		Alpha:             alpha,
		Iterations:        iterations,
		Seed:              seed,
		Workers:           workers,
		CandidateRecall:   candidateRecall,
		CandidateControls: candidateControls,
	})
	if err != nil {
		return Config{}, fmt.Errorf("fasthnsw: invalid persisted config: %w", err)
	}
	return cfg, nil
}

func readRequiredInt(r io.Reader, name string, max int) (int, error) {
	value, err := readUint32(r)
	if err != nil {
		return 0, fmt.Errorf("fasthnsw: read %s: %w", name, err)
	}
	return persistedPositiveInt(value, name, max)
}

func validatePersistableIndex(idx *Index) error {
	if !idx.graphReady {
		return ErrIndexNotBuilt
	}
	if idx.count <= 0 {
		return fmt.Errorf("fasthnsw: index has no vectors")
	}
	if idx.dim <= 0 {
		return fmt.Errorf("fasthnsw: index dimension must be positive")
	}
	if idx.count > maxPersistedCount {
		return fmt.Errorf("fasthnsw: vector count %d exceeds persistence limit %d", idx.count, maxPersistedCount)
	}
	if idx.dim > maxPersistedDimension {
		return fmt.Errorf("fasthnsw: dimension %d exceeds persistence limit %d", idx.dim, maxPersistedDimension)
	}
	if _, err := checkedVectorValueCount(idx.count, idx.dim); err != nil {
		return err
	}
	if len(idx.vectors) != idx.count*idx.dim {
		return fmt.Errorf("fasthnsw: flat vector length %d does not match count*dim %d", len(idx.vectors), idx.count*idx.dim)
	}
	if len(idx.levels) != idx.count {
		return fmt.Errorf("fasthnsw: level count %d does not match vector count %d", len(idx.levels), idx.count)
	}
	if idx.entryPoint < 0 || idx.entryPoint >= idx.count {
		return fmt.Errorf("fasthnsw: entry point %d out of range", idx.entryPoint)
	}
	if idx.maxLayer < 0 || idx.maxLayer >= maxPersistedLayers {
		return fmt.Errorf("fasthnsw: max layer %d out of persistence range", idx.maxLayer)
	}
	if len(idx.layers) != idx.maxLayer+1 {
		return fmt.Errorf("fasthnsw: layer count %d does not match max layer %d", len(idx.layers), idx.maxLayer)
	}
	if err := validatePersistedConfigForWrite(idx.cfg); err != nil {
		return err
	}
	return validateLayerAdjacency(idx.cfg, idx.levels, idx.layers, idx.count, idx.entryPoint, idx.maxLayer)
}

func validateLoadedIndex(idx *Index, maxSeenLevel int) error {
	if idx.entryPoint < 0 || idx.entryPoint >= idx.count {
		return fmt.Errorf("fasthnsw: entry point %d out of range", idx.entryPoint)
	}
	if idx.levels[idx.entryPoint] != idx.maxLayer {
		return fmt.Errorf("fasthnsw: entry point level %d does not match max layer %d", idx.levels[idx.entryPoint], idx.maxLayer)
	}
	if maxSeenLevel != idx.maxLayer {
		return fmt.Errorf("fasthnsw: max level in assignments %d does not match max layer %d", maxSeenLevel, idx.maxLayer)
	}
	return validateLayerAdjacency(idx.cfg, idx.levels, idx.layers, idx.count, idx.entryPoint, idx.maxLayer)
}

func validateLayerAdjacency(cfg Config, levels []int, layers [][][]int, count int, entryPoint int, maxLayer int) error {
	if len(layers) != maxLayer+1 {
		return fmt.Errorf("fasthnsw: layer count %d does not match max layer %d", len(layers), maxLayer)
	}
	for layer, adjacency := range layers {
		if len(adjacency) != count {
			return fmt.Errorf("fasthnsw: layer %d adjacency length %d does not match vector count %d", layer, len(adjacency), count)
		}
		maxDegree := layerMaxDegree(cfg, layer)
		for sourceID, neighbors := range adjacency {
			if levels[sourceID] < layer {
				if len(neighbors) != 0 {
					return fmt.Errorf("fasthnsw: layer %d node %d is below layer but has neighbors", layer, sourceID)
				}
				continue
			}
			if len(neighbors) > maxDegree {
				return fmt.Errorf("fasthnsw: layer %d node %d has degree %d above bound %d", layer, sourceID, len(neighbors), maxDegree)
			}
			seen := make(map[int]bool, len(neighbors))
			for _, neighborID := range neighbors {
				if neighborID < 0 || neighborID >= count {
					return fmt.Errorf("fasthnsw: layer %d node %d neighbor %d out of range", layer, sourceID, neighborID)
				}
				if neighborID == sourceID {
					return fmt.Errorf("fasthnsw: layer %d node %d contains self-neighbor", layer, sourceID)
				}
				if levels[neighborID] < layer {
					return fmt.Errorf("fasthnsw: layer %d node %d links to node %d below layer", layer, sourceID, neighborID)
				}
				if seen[neighborID] {
					return fmt.Errorf("fasthnsw: layer %d node %d contains duplicate neighbor %d", layer, sourceID, neighborID)
				}
				seen[neighborID] = true
			}
		}
	}
	if levels[entryPoint] < maxLayer {
		return fmt.Errorf("fasthnsw: entry point %d is below max layer %d", entryPoint, maxLayer)
	}
	return nil
}

func validatePersistedConfigForWrite(cfg Config) error {
	if _, err := normalizeConfig(cfg); err != nil {
		return fmt.Errorf("fasthnsw: invalid index config: %w", err)
	}
	if cfg.M <= 0 || cfg.M > maxPersistedDegree {
		return fmt.Errorf("fasthnsw: M %d out of persistence range", cfg.M)
	}
	if cfg.EfConstruction <= 0 || cfg.EfConstruction > maxPersistedDegree {
		return fmt.Errorf("fasthnsw: EfConstruction %d out of persistence range", cfg.EfConstruction)
	}
	if cfg.K0 <= 0 || cfg.K0 > maxPersistedDegree {
		return fmt.Errorf("fasthnsw: K0 %d out of persistence range", cfg.K0)
	}
	if cfg.CandidateK <= 0 || cfg.CandidateK > maxPersistedDegree {
		return fmt.Errorf("fasthnsw: CandidateK %d out of persistence range", cfg.CandidateK)
	}
	if cfg.ConstructionL <= 0 || cfg.ConstructionL > maxPersistedDegree {
		return fmt.Errorf("fasthnsw: ConstructionL %d out of persistence range", cfg.ConstructionL)
	}
	if cfg.Iterations <= 0 || cfg.Iterations > maxPersistedDegree {
		return fmt.Errorf("fasthnsw: Iterations %d out of persistence range", cfg.Iterations)
	}
	if cfg.Workers <= 0 || cfg.Workers > maxPersistedDegree {
		return fmt.Errorf("fasthnsw: Workers %d out of persistence range", cfg.Workers)
	}
	if cfg.CandidateControls <= 0 || cfg.CandidateControls > maxPersistedDegree {
		return fmt.Errorf("fasthnsw: CandidateControls %d out of persistence range", cfg.CandidateControls)
	}
	return nil
}

func layerMaxDegree(cfg Config, layer int) int {
	if layer == 0 {
		return 2 * cfg.M
	}
	return cfg.M
}

func checkedVectorValueCount(count int, dim int) (int, error) {
	if count <= 0 || dim <= 0 {
		return 0, fmt.Errorf("fasthnsw: count and dimension must be positive")
	}
	if count > maxPersistedVectorValues/dim {
		return 0, fmt.Errorf("fasthnsw: vector storage %dx%d exceeds persistence limit", count, dim)
	}
	return count * dim, nil
}

func persistedPositiveInt(value uint32, name string, max int) (int, error) {
	if value == 0 {
		return 0, fmt.Errorf("fasthnsw: %s must be positive", name)
	}
	return persistedNonNegativeInt(value, name, max)
}

func persistedNonNegativeInt(value uint32, name string, max int) (int, error) {
	if value > uint32(max) {
		return 0, fmt.Errorf("fasthnsw: %s %d exceeds limit %d", name, value, max)
	}
	return int(value), nil
}

func writeUint32(w io.Writer, value uint32) error {
	var buf [4]byte
	persistenceOrder.PutUint32(buf[:], value)
	return writeFull(w, buf[:])
}

func readUint32(r io.Reader) (uint32, error) {
	var buf [4]byte
	if err := readFull(r, buf[:]); err != nil {
		return 0, err
	}
	return persistenceOrder.Uint32(buf[:]), nil
}

func writeInt32(w io.Writer, value int32) error {
	return writeUint32(w, uint32(value))
}

func readInt32(r io.Reader) (int32, error) {
	value, err := readUint32(r)
	return int32(value), err
}

func writeUint64(w io.Writer, value uint64) error {
	var buf [8]byte
	persistenceOrder.PutUint64(buf[:], value)
	return writeFull(w, buf[:])
}

func readUint64(r io.Reader) (uint64, error) {
	var buf [8]byte
	if err := readFull(r, buf[:]); err != nil {
		return 0, err
	}
	return persistenceOrder.Uint64(buf[:]), nil
}

func writeInt64(w io.Writer, value int64) error {
	return writeUint64(w, uint64(value))
}

func readInt64(r io.Reader) (int64, error) {
	value, err := readUint64(r)
	return int64(value), err
}

func writeFull(w io.Writer, data []byte) error {
	for len(data) > 0 {
		written, err := w.Write(data)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		data = data[written:]
	}
	return nil
}

func readFull(r io.Reader, data []byte) error {
	_, err := io.ReadFull(r, data)
	return err
}
