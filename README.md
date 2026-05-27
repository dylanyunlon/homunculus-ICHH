# Homunculus — Pipelined Index Construction for Billion-Scale Vectors on Heterogeneous Hardware

> *"Es leuchtet! Seht! — Nun lässt sich wirklich hoffen..."*
> — Wagner, Faust II Act 2

Build billion-scale vector indices through a 3-stage heterogeneous pipeline:
CPU (preprocess) → A6000 (coarse graph via FastHNSW) → H100 (refinement via VSAG).

## Upstream

| Directory | Origin | Role |
|-----------|--------|------|
| `upstream/vsag` | [antgroup/vsag](https://github.com/antgroup/vsag) | Graph-based ANN index engine |
| `upstream/fasthnsw` | [cryo-zd/fasthnsw](https://github.com/cryo-zd/fasthnsw) | Fast HNSW construction path |

## Pipeline

```
Stage 1 (CPU)         Stage 2 (A6000×2)      Stage 3 (H100)
┌──────────────┐      ┌──────────────┐       ┌──────────────┐
│Data ingestion│ ───► │Coarse HNSW   │ ────► │Fine-grained  │
│Normalization │      │(FastHNSW)    │       │graph refine  │
│Partitioning  │      │GDDR6 48GB×2  │       │(VSAG) HBM3   │
└──────────────┘      └──────────────┘       └──────────────┘
```
