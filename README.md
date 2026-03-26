# eLLM

## Description
eLLM provides million-token inference on CPU-only machines

## Operators taxonomy

`src/operators` now exposes 5 semantic categories to make operator responsibilities easier to understand and maintain:

- `operators::linear`: dense linear algebra (`MatMul`, `MatMul3`, `MatMulAdd`, `Attention`)
- `operators::routing`: token/expert routing and candidate selection (`MatMulTopK`, `ExpertsSoftmaxNorm`, `TopKSoftmax`)
- `operators::expert`: expert-network compute and merge (`ExpertsMatMulSilu`, `ExpertsMatMulDown`, `ExpertsMergeAdd`)
- `operators::transform`: elementwise + normalization transforms (`AddZipMap`, `SiluMulZipMap`, `ComplexZipMap`, `RMSMap`, `LookupRMSMap`, `AddRMSZipMap`)
- `operators::movement`: state/data movement operators (`LiftVector`)
- `operators::traits`: centralized trait definitions used by all operator categories

### Classification criteria

- **Primary math semantics first**: grouped by what the operator computes, not by where SIMD/scalar kernels live.
- **Pipeline role second**: routing, expert execution, transform, and movement are separated by inference-stage responsibility.
- **Stable call-site ergonomics**: runtime code can import from semantic buckets without needing knowledge of internal file layout.
- **Trait centralization**: all trait definitions are merged under `src/operators/traits` for unified maintenance.

## Docs

Selected design notes live under [`docs/`](docs/):

- [`docs/matmul.md`](docs/matmul.md): high-performance matrix multiplication principles and how they map to this repository
- [`docs/attention.md`](docs/attention.md): attention scheduling and static parallel decomposition
- [`docs/minimal_model_abstraction.md`](docs/minimal_model_abstraction.md): minimal model abstraction strategy


