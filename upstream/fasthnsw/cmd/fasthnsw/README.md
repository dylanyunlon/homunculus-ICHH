# Build and Query

The optional smoke CLI uses a simple text vector format: one vector per line,
whitespace-separated `float32` values, with blank lines and lines starting with
`#` ignored. Row numbers are used as vector ids.

```sh
go run ./cmd/fasthnsw build \
  -input vectors.txt \
  -output index.fhnsw \
  -dim 128 \
  -metric l2 \
  -workers 4

go run ./cmd/fasthnsw query \
  -index index.fhnsw \
  -queries queries.txt \
  -k 10 \
  -ef 64
```

# Validation and Benchmarks

> Use `-algorithm hnsw` on the same validation command to build a standard incremental HNSW baseline for internal construction-speed comparisons. The baseline is wired into the CLI only; it is not part of the public Go API.

Generated clustered or uniform validation runs without external files:

```sh
go run ./cmd/fasthnsw validate \
  -algorithm fasthnsw \
  -dataset clustered \
  -vectors 1000 \
  -query-count 100 \
  -dim 32 \
  -k 10 \
  -ef 64
```

ANN-Benchmarks HDF5 files should contain `train`, `test`, and `neighbors`
datasets:

```sh
go run ./cmd/fasthnsw validate \
  -algorithm fasthnsw \
  -dataset hdf5 \
  -input sift-128-euclidean.hdf5 \
  -metric l2 \
  -limit-queries 1000 \
  -k 10 \
  -ef 64
```

Use `-limit-base` only when the provided ground truth was produced for the
same truncated base set; otherwise Recall@k is not meaningful and validation
will reject out-of-range ground-truth ids.

SIFT-style raw files are also supported:

```sh
go run ./cmd/fasthnsw validate \
  -algorithm fasthnsw \
  -dataset fvecs \
  -base sift_base.fvecs \
  -queries sift_query.fvecs \
  -truth sift_groundtruth.ivecs \
  -metric l2 \
  -k 10 \
  -ef 64
```

Local Go benchmarks include distance throughput, candidate acquisition, layer
construction, full build, and search latency:

```sh
go test -bench=. -benchmem ./...
```
