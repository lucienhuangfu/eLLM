# eLLM Operator Reference

Lookup material for the `create_operator` skill. The workflow lives in [SKILL.md](../SKILL.md); this file covers catalogs, signatures, and on-request extension patterns.

## Operator Catalog

| Family | Operators | Compute traits |
|--------|-----------|----------------|
| attention/ | `Attention<T>` | `AttentionTrait` |
| elementwise/ | `AddZipMap`, `ComplexZipMap` (`ZipMapTrait`); `SigmoidMap` (`MapTrait`); `SiluMulZipMap` (fused SiLU×mul) | map / zip |
| expert/ | `ExpertMatMulDown`, `ExpertMatMulSilu`, `ExpertMergeAdd`, `ExpertRouting`, `ExpertTopkNorm` | `ExpertsDownTrait`, `ExpertsSiluTrait`, `MoeMergeTrait`, `ExpertsTopkNormTrait` |
| matmul/ | `MatMul`, `MatMul3` (head-by-head K/Q/V), `MatMulAdd` (fused GEMM+residual), `MatMulSigmoid`, `MatMulTopK` | `MatMul*Trait` |
| normalization/ | `RMSMap` (canonical simple example), `AddRMSZipMap` (fused), `LookupRMSMap` (embedding+norm) | `MapTrait` |
| softmax/ | `ExpertsSoftmaxNorm`, `TopKSoftmax` | `SoftmaxTrait`, `TopKSoftmaxTrait` |

Family choice: per-element math → `elementwise/`, GEMM-family → `matmul/`, norm → `normalization/`, MoE compute → `expert/`, routing/topk → `softmax/` or `expert/`.

Helpers (top level): `operator.rs` (dispatch enum), `assign.rs` (`assign()` even thread partition, `assign_kqv_tile()` for merged K/Q/V), `send_sync_ptr.rs` (`ConstPtr`/`MutPtr`/`SharedMut`), `fake_echo.rs` / `lift_vector.rs` (minimal reference operators).

## Traits (`src/operators/traits/`)

```rust
// map.rs — the two simplest primitives
pub trait MapTrait<T>    { fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize); }
pub trait ZipMapTrait<T> { fn compute(&self, input_ptr1: *const T, input_ptr_2: *const T, output_ptr: *mut T); }
```

Other trait files: `linear.rs` (`AttentionTrait`, `MatMulTrait`, `MatMulAddTrait`, `MatMulSigmoidTrait`, `MatMulkqvTrait`), `expert.rs` (`ExpertsDownTrait`, `ExpertsSiluTrait`, `MoeMergeTrait`), `softmax.rs` (`MatMulTopKTrait`, `TopKSoftmaxTrait`, `SoftmaxTrait`, `ExpertsTopkNormTrait`). New primitive → add trait there and re-export in `mod.rs`.

## `run` Signature Families

Dispatch entry (called once per worker per round): `Operator::run(prefill_size, decode_size, lift_size, total_size, cpu_num, thread_id, computing_slices, slot_list)`. Individual operators take a subset:

| Signature | Used by | Dispatch |
|-----------|---------|----------|
| `(prefill, decode, total, cpu_num, thread_id)` | AddRMSZipMap, AddZipMap, MatMul, MatMulSigmoid, ExpertsSoftmaxNorm, MatMulTopK | `run_simple!` |
| `(prefill, decode, total, lift, cpu_num, thread_id)` | RMSMap, MatMulAdd, ExpertsMatMulDown/Silu/MergeAdd/TopkNorm | explicit |
| `(total, cpu_num, thread_id)` | SigmoidMap | explicit |
| `(total, computing_slices, cpu_num, thread_id)` | Attention, LiftVector | explicit |
| `(total, cpu_num, thread_id, computing_slices)` | LookupRMSMap | explicit |
| slot-aware (`slot_list`) | TopKSoftmax, FakeEcho | explicit |

Semantics: `total_size = prefill_size + decode_size`; `lift_size` is the decode-side row count for decode-only operators (`decode_only_flag`); `computing_slices`/`slot_list` only for per-sequence/per-slot state. `kind()` must return the variant name as `&'static str`.

## `mod.rs` Re-export Groups

Declare the file in its family module, then re-export by role:

| Group | Role |
|-------|------|
| `transform` | elementwise / normalization maps and zips |
| `linear` | GEMM-family and attention |
| `routing` | MoE routing / topk / softmax |
| `expert` / `expert_imports` | MoE expert compute (`#[allow(non_snake_case)]` aliases like `ExpertMatMulDown as ExpertsMatMulDown`) |
| `movement` | memory movement |
| `testing` | test-only operators |

## On Request: Fill Compute Logic

Write it inline in the trait impl (plain scalar Rust):

```rust
impl<T: Sqrt> MapTrait<T> for MyOp<T> {
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize) {
        // scalar loop over [0, length)
    }
}
```

Follow `RMSMap`'s specialization pattern only when needed: generic `default fn` fallback + per-dtype impls (`f16`/`f32`/`f64`); SIMD impls gate on `#[cfg(all(target_arch = "x86_64", target_feature = "..."))]` with a scalar `#[cfg(not(...))]` fallback. Numeric bounds (`Sqrt`, `Exp`, `Sigmoid`, `FromNumber`, `NegInfinity`) come from `crate::num_traits`.

## On Request: Extract Kernel

Move the computation into `crate::kernel` (`kernel::scalar` fallback, optional `kernel::x86_64` SIMD), leaving the operator file as scheduling + dispatch — same shape as `RMSMap` calling `kernel::scalar::rms_norm::rms_norm`.

## On Request: Tensor API Wiring

Operators enter the graph only through `Tensor` builder methods (`src/tensor/ops.rs`, `src/tensor/matmul.rs`); transformer code never constructs operators directly:

```rust
pub fn my_op(&self, /* weights, config, scope_name */) -> Self {
    let output_tensor = Self::output_tensor(self.shape.clone(), &scope_name);
    let operator = Operator::MyOp(MyOp::new(self.data, output_tensor.data, /* ... */));
    Self::enqueue(operator);
    output_tensor
}
```

Transformer layers call these with a `scope_name` like `"{layer}.post_attention_layernorm"`.

## On Request: Alignment Scaffold

```bash
python alignment/scripts/create_new_operator.py <op_name>   # Windows: py
```

Creates `alignment/<op_name>/`: `generate_hf_<op>.py` (reference → `python_<op>_*.npy`), `<op>_alignment_test.rs` (Rust → `rust_<op>_output.npy`, register as `[[bin]]` in Cargo.toml), `test_<op>_alignment.py` (thresholds: `max_abs < 1e-5`, `mean_abs < 1e-6`, `cosine > 0.999999`). Then hand off to the `align_operator` skill.

## Unit Test Pattern

```rust
#[test]
fn test_my_op() {
    let (batch_size, hidden_size) = (10, 18);
    let input_data: Vec<f32> = (1..=18).cycle().take(180).map(|x| x as f32).collect();
    let mut output_data = vec![0.0f32; batch_size * hidden_size];
    let operator = MyOp::new(input_data.as_ptr(), output_data.as_mut_ptr(), hidden_size);
    let thread_num = 4;
    for i in 0..thread_num {
        operator.run(batch_size, 0, batch_size, batch_size, thread_num, i);
    }
    // while compute is empty: only assert no panic / partition coverage
    // after compute lands: assert_ulps_eq!(output_data[..], expected[..], max_ulps = 4);
}
```

Deterministic inputs (sequential / zeros / ones); loop `thread_id` over `thread_num` so slices cover the tensor exactly once.

## Pitfalls

- Allocating in `run()` — violates the static-graph contract; pre-allocate at construction (see `attention/scratch.rs`).
- Bare raw pointers in the struct — `Operator<T>` fails `Send`/`Sync`; use `ConstPtr`/`MutPtr`.
- Hand-rolled thread division — use `assign()`.
- Decode-only operator reading `total_size` — honor `decode_only_flag` and use `lift_size`.
- Enum variant added but `kind()` arm forgotten.
- Hardcoded offsets instead of strides passed via the constructor.
