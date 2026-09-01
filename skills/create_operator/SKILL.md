---
name: create_operator
description: Create a new Rust operator in the eLLM inference engine under src/operators. Use when adding a new computation operator (map, zip-map, matmul, attention, softmax, routing, or expert/MoE operator). Default scope is the operator skeleton with an empty compute method (TODO); compute logic, crate::kernel extraction, Tensor API wiring, and alignment scaffolding are added only when the user explicitly requests them.
---

# Create Operator

Create an operator skeleton under `src/operators/`, register it in the dispatch enum, and leave `compute` empty.

## Scope

| Deliverable | Default |
|-------------|---------|
| Operator file: struct + `new` + `run` (thread partition) + empty `compute` (TODO) | ✅ |
| Registration in `mod.rs` and the `Operator<T>` enum | ✅ |
| Minimal unit test (construction, partition, no panic) | ✅ |
| Compute logic (inline scalar) | explicit request |
| Compute kernel extracted into `crate::kernel` | explicit request |
| Tensor API wiring (`src/tensor/*` builder + `enqueue`) | explicit request |
| Alignment scaffold (`alignment/scripts/create_new_operator.py`) | explicit request |

## Static Graph Contract

eLLM is a static-graph engine (unlike PyTorch eager): the graph is built once at model load — all tensors pre-allocated, all operators enqueued — and `ExecutorPool` replays the same queue every round. Therefore:

- Construction captures pointers to pre-allocated buffers; the operator owns nothing.
- `run()` is allocation-free and processes only this thread's slice.
- Per-round dynamism arrives via `run()` size parameters, never via graph rebuilding.

## Skeleton

```rust
// src/operators/<family>/<op_name>.rs
use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::MapTrait;

#[derive(Clone)]
pub struct MyOp<T> {
    ptr1: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    hidden_size: usize,
}

impl<T> MyOp<T> {
    pub fn new(ptr1: *const T, output_ptr: *mut T, hidden_size: usize) -> Self {
        Self { ptr1: ConstPtr { ptr: ptr1 }, output_ptr: MutPtr { ptr: output_ptr }, hidden_size }
    }

    pub fn run(&self, _prefill_size: usize, _decode_size: usize, total_size: usize,
               _lift_size: usize, thread_num: usize, thread_id: usize) {
        if let Some((begin, end)) = assign(total_size, thread_num, thread_id) {
            for index in begin..end {
                // TODO: process row `index` via self.compute(...)
            }
        }
    }
}

impl<T> MapTrait<T> for MyOp<T> {
    fn compute(&self, _input_ptr: *const T, _output_ptr: *mut T, _length: usize) {
        // TODO: compute logic, filled in later
    }
}
```

Rules:

- Wrap raw pointers in `ConstPtr` / `MutPtr` (keeps the struct `Send + Sync`).
- Derive `Clone`; construction is cheap (pointer + config copies).
- Partition work with `assign()` only; never hand-roll thread division.
- `run()` allocates nothing; scratch must be a pre-allocated buffer captured at construction.
- Implement `compute` through the trait matching the operator family (see the reference).

## Register

1. Declare the file in the family module of `src/operators/mod.rs` (+ role re-export group if applicable).
2. In `src/operators/operator.rs`, three places: enum variant (alphabetical), `match` arm in `run()` (`run_simple!` only when the signature is exactly `(prefill_size, decode_size, total_size, cpu_num, thread_id)`), arm in `kind()`.

## Verify

```bash
cargo build && cargo test <op_name>
```

Unit test stays minimal while `compute` is empty: construct, run all threads in a loop, assert no panic / partition coverage. Add numerical assertions (`approx::assert_ulps_eq!`, `max_ulps = 4`) after compute lands.

## Reference

See [references/operator-spec.md](references/operator-spec.md) for the operator catalog, traits, `run` signature table, re-export groups, on-request extension patterns, and pitfalls.
