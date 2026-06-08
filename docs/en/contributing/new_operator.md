# Adding a New Operator

This guide walks through the process of implementing, integrating, and testing
a new operator in eLLM.

---

## 1. Understand the Operator Model

An operator in eLLM is a unit of computation that is placed into the operator
queue by `initialize_serving_resources()` and executed by `ServingRunner` each
scheduling round.

Key traits and types:

- `Operator<T>` enum — the dispatch enum in `src/operators/operator.rs`
- `run(thread_id, thread_num, task: &ScheduleTask)` — called once per scheduling round per thread
- `ScheduleTask` — carries `prefill_list`, `decode_list`, and counts for the current round

Operators should be stateless with respect to requests. All mutable request
state lives in `BatchSequence` and `Vec<SequenceState>`.

---

## 2. Create the Operator File

Add a new file under the appropriate `src/operators/` subdirectory.

Example skeleton for a simple elementwise operator:

```rust
// src/operators/elementwise/my_op.rs

use crate::runtime::scheduling::types::ScheduleTask;

pub struct MyOp {
    // configuration captured at construction
}

impl MyOp {
    pub fn new(/* ... */) -> Self {
        MyOp { /* ... */ }
    }

    pub fn run(&self, thread_id: usize, thread_num: usize, task: &ScheduleTask) {
        // Use task.decode_list or task.prefill_list to determine which tokens
        // to process this round.
        //
        // Use assign(total, thread_num, thread_id) to split work across threads.
    }
}
```

Expose it in the parent `mod.rs`:

```rust
pub mod my_op;
pub use my_op::MyOp;
```

---

## 3. Add to the Operator Enum

In `src/operators/operator.rs`, add a variant:

```rust
pub enum Operator<T> {
    // ... existing variants ...
    MyOp(MyOp),
}

impl<T> Operator<T> {
    pub fn run(&self, thread_id: usize, thread_num: usize, task: &ScheduleTask) {
        match self {
            // ... existing arms ...
            Operator::MyOp(op) => op.run(thread_id, thread_num, task),
        }
    }
}
```

---

## 4. Push the Operator onto the Queue

In `src/serving/resources.rs` (or wherever the operator queue is assembled),
push the new operator:

```rust
operator_queue.push(Operator::MyOp(MyOp::new(/* ... */)));
```

The queue is executed in order by `ServingRunner` every round.

---

## 5. Write an Alignment Test

The recommended validation workflow:

1. Write a Python script (under `alignment/<op_name>/`) that computes the
   expected output using a reference implementation (NumPy, PyTorch, etc.) and
   saves it as `.npy` files.
2. Write a Rust alignment binary that loads the same inputs, runs the operator,
   and compares against the saved `.npy` outputs.

See `alignment/silu_mul/` and `alignment/rope/` for worked examples.

Alignment binary entry point in `Cargo.toml`:

```toml
[[bin]]
name = "my_op_alignment"
path = "alignment/my_op/my_op_alignment.rs"
```

Run the comparison:

```bash
python alignment/my_op/generate_data.py
cargo run --bin my_op_alignment
```

---

## 6. Naming Conventions

| Thing | Convention |
|-------|-----------|
| Operator struct | `PascalCase` (e.g., `SiluMulZip`) |
| Operator file | `snake_case.rs` (e.g., `silu_mul_zip.rs`) |
| Alignment directory | `alignment/<op_name>/` |
| Test input files | `python_<op_name>_<role>.npy` |

---

## 7. References

- [FakeEcho](../operator/fake_echo.md) — simplest possible operator, good reference
- [LiftVector](../operator/left_vector.md) — minimal memory-movement operator
- [Attention](../operator/attention.md) — full static parallel operator
- [HuggingFace Alignment](../reference/hf_alignment.md) — golden-file testing workflow
