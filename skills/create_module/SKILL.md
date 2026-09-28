---
name: create_module
description: Create a new transformer module in the eLLM inference engine under src/transformer. Use when adding a new model building block such as an attention variant, an MLP/FFN variant, a router, or a decoder-layer component. Default scope is the module skeleton (struct + new + stubbed forward), mod.rs registration, its tensor-name struct in names.rs, and a minimal unit test; forward graph composition via the Tensor API, DecoderLayer/Config wiring, multi-file directory split, and alignment scaffolding are added only when the user explicitly requests them.
---

# Create Module

Create a transformer module skeleton under `src/transformer/`, register it in `mod.rs`, add its tensor-name struct in `names.rs`, and leave `forward` as a `todo!()` stub.

## Scope

| Deliverable | Default |
|-------------|---------|
| Module file: struct + `new` + stubbed `forward` (`todo!`) | ✅ |
| Declaration in `src/transformer/mod.rs` | ✅ |
| Tensor-name struct in `src/transformer/names.rs` | ✅ |
| Minimal unit test (construction, forward shape, no panic) | ✅ |
| Forward graph composition via Tensor API | explicit request |
| DecoderLayer wiring (`AttentionBlock`/`FfnBlock` enum + match arms) | explicit request |
| Config kind enum extension (`AttentionKind`/`FfnKind` in `model_family/config`) | explicit request |
| Multi-file split into `<module>/` directory (sparse_moe pattern) | explicit request |
| Alignment scaffold (`alignment/`) | explicit request |

## Static Graph Contract

eLLM is a static-graph engine (unlike PyTorch eager): the graph is built once at model load — `forward` of every module is called a single time, each Tensor API call pre-allocates its output and enqueues an operator — then `ExecutorPool` replays the same queue every round. Therefore:

- A module owns its weight tensors (`Tensor::zeros(shape, name)`); activations are never stored in the struct.
- `new()` receives HF-compatible tensor names from `names.rs`; weight loading fills those names with safetensors data later.
- `forward()` composes Tensor builder methods only — it never constructs operators directly.
- Per-round dynamism arrives via `decode_only_flag` and size parameters, never via graph rebuilding.

## Skeleton

```rust
// src/transformer/<module_name>.rs
use std::ops::{AddAssign, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, FromNumber, NegInfinity, Sigmoid, Sqrt};
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::names::MyModuleTensorNames;

#[derive(Clone)]
pub struct MyModule<T>
where
    T: Copy + PartialOrd,
{
    some_weight: Tensor<T>,
    scope_name: String,
}

impl<T> MyModule<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + FromNumber
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    pub fn new(hidden_size: usize, names: MyModuleTensorNames) -> Self {
        Self {
            some_weight: Tensor::zeros(vec![hidden_size, hidden_size], names.some_proj),
            scope_name: names.scope,
        }
    }

    pub fn forward(
        &self,
        _hidden_states: &Tensor<T>,
        _residual: &Tensor<T>,
        _decode_only_flag: bool,
        _tensor_name: String,
    ) -> Tensor<T> {
        // TODO: compose Tensor API calls (matmul, rms, silu_mul, ...)
        todo!("forward graph composition")
    }
}
```

Rules:

- Weights are created with `Tensor::zeros(shape, name)`; the name comes from the `names.rs` struct and must match the HF safetensors key so weight loading fills it.
- Tensor names must match the `mem_pool.rs` `REGEX_SET` patterns (e.g. `model.layers.N.<...>`, `*.weight`, `model.*.output`), otherwise allocation panics.
- Keep the generic bound list exactly as above; numeric bounds (`Exp`, `Sigmoid`, `Sqrt`, ...) come from `crate::num_traits`.
- `forward` signature convention: `(&self, hidden_states, residual, decode_only_flag, tensor_name) -> Tensor<T>`; drop or add arguments to fit the module role (see `Attention` vs `DenseMlp`).
- Output tensor names are built from `self.scope_name` with `format!("{}.<suffix>", self.scope_name)`.
- Derive `Clone` when possible — `Tensor` is a cheap handle (pointer + shape + name).

## Register

1. Declare the file in `src/transformer/mod.rs` (`pub mod <module_name>;`, alphabetical).
2. In `src/transformer/names.rs`: add a `#[derive(Debug, Clone)]` `<Module>TensorNames` struct (`scope` + one field per weight). If the module lives under a decoder layer, generate it inside `layer_tensor_names()` following HF naming, e.g. `model.layers.{i}.mlp.gate_proj.weight`.

## Verify

```bash
cargo build && cargo test <module_name>
```

Unit test stays minimal while `forward` is stubbed: construct the module, assert weight shapes. After `forward` lands, call it and assert the output shape, then drain the operator queue (see the reference for the queue-execution pattern).

## Reference

See [references/module-spec.md](references/module-spec.md) for the module catalog, tensor-name structs, Tensor API catalog, `DecoderLayer` wiring, on-request extension patterns, and pitfalls.
