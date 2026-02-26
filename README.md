# eLLM

## Description
eLLM provides million-token inference on CPU-only machines

## Ops operator taxonomy

`src/ops` now exposes 5 semantic categories to make operator responsibilities easier to understand and maintain:

- `ops::linear`: dense linear algebra (`MatMul`, `MatMul3`, `MatMulAdd`, `Attention`)
- `ops::routing`: token/expert routing and candidate selection (`MatMulTopK`, `ExpertsSoftmaxNorm`, `TopKSoftmax`)
- `ops::expert`: expert-network compute and merge (`ExpertsMatMulSilu`, `ExpertsMatMulDown`, `ExpertsMergeAdd`)
- `ops::transform`: elementwise + normalization transforms (`AddZipMap`, `SiluMulZipMap`, `ComplexZipMap`, `RMSMap`, `LookupRMSMap`, `AddRMSZipMap`)
- `ops::movement`: state/data movement operators (`LiftVector`)
- `ops::traits`: centralized trait definitions used by all operator categories

### Classification criteria

- **Primary math semantics first**: grouped by what the operator computes, not by where SIMD/scalar kernels live.
- **Pipeline role second**: routing, expert execution, transform, and movement are separated by inference-stage responsibility.
- **Stable call-site ergonomics**: runtime code can import from semantic buckets without needing knowledge of internal file layout.
- **Trait centralization**: all trait definitions are merged under `src/ops/traits` for unified maintenance.


