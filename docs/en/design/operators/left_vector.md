# LiftVector: Copy the Last Token Vector into a Compact Decode Buffer

This document explains the `LiftVector` operator implemented in [`src/operators/left_vector.rs`](../../../src/operators/left_vector.rs).

In the codebase, it is exposed through [`Tensor::lift_vector()`](../../../src/runtime/tensor.rs) and is usually triggered from the attention forward path when `decode_only_flag` is enabled.

---

# 1. What This Operator Does

`LiftVector` is not a math-heavy operator. It is a memory movement operator.

Its job is:

* Read the last token vector from each selected `SequenceSlice`
* Copy that vector into a compact output region
* Keep the output region densely packed in decode order

In other words, it "lifts" the final token representation of each slice from the original tensor layout into a contiguous decode buffer.

The operator works on a single tensor buffer in place:

* Source data lives in the original `hidden_states` storage
* Destination data is written back into the same storage at earlier positions
* The copy is done with `ptr::copy_nonoverlapping`, so the source and destination ranges must not overlap in a way that violates non-overlap assumptions

---

# 2. Why It Exists

In the decode path, not every token in a slice is equally interesting. The model usually needs the final token representation of each slice as the summary token for downstream work.

This operator exists to solve that exact layout problem:

* The scheduler already knows which slices are active
* Each `SequenceSlice` carries its own token range
* `LiftVector` turns the "last token of each slice" into a compact, front-packed buffer

So the operator is best understood as:

```text
extract the last token vector for each active slice
and pack those vectors densely by decode order
```

---

# 3. Data Model

The operator consumes a `decode_list: &[SequenceSlice]`.

Each `SequenceSlice` provides:

* `token_start_index`
* `length`
* `last_token_flag`
* `batch_index`
* `sequence_index`

Only three of them matter directly to `LiftVector`:

* `token_start_index`
* `length`
* `last_token_flag`

The output slot is determined by the slice position in the assigned thread range, not by `batch_index` or `sequence_index`.

The tensor width used by the operator is `self.length` inside `LiftVector`, which comes from `Tensor::lift_vector()` as `self.shape[1]`.

So if the underlying tensor is logically:

```text
[token_count, hidden_size]
```

then `hidden_size` is the vector length copied for each token.

---

# 4. Execution Flow

## 4.1 Static Thread Split

`LiftVector::run()` uses the same continuous range split style as other operators in this repository:

```rust
let Some((begin, end)) = assign(total_tokens, thread_num, thread_id) else {
    return;
};
```

This means:

* The total number of slices is split into continuous ranges
* Each thread processes only its assigned slice interval
* If there is no work for the current thread, it returns immediately

The split is over `decode_list.len()`, not over the hidden-size dimension inside each vector.

## 4.2 Per-Slice Copy Logic

For every slice in the assigned range:

1. Skip the slice if `last_token_flag` is false
2. Compute the source token index:

   ```text
   source_token_index = token_start_index + length - 1
   ```

3. Compute the destination slot:

   ```text
   destination_index = begin + offset
   ```

4. Copy `self.length` elements from the source token vector to the destination slot

So the essential mapping is:

```text
source: last token of the slice
dest:   compact slot in decode output order
```

---

# 5. Important Semantics

## 5.1 `last_token_flag`

`last_token_flag` is the guard that says whether this slice should participate in lifting.

If it is `false`:

* The slice is skipped
* No copy occurs
* The destination slot for that offset remains untouched by this operator

This makes the operator compatible with layouts where only certain slices are the final token contributors in a round.

## 5.2 Destination Packing

The destination slot is computed with `begin + offset`, where `offset` is the index inside the thread-owned range.

That gives the operator two useful properties:

* Each thread writes a compact, continuous destination segment
* The final output stays ordered by slice iteration order

This is a "compact pack" operation, not a scatter.

## 5.3 Non-Overlapping Copy Assumption

The implementation uses `ptr::copy_nonoverlapping`.

That means the code assumes:

* The source token vector and destination slot do not overlap in a way that breaks non-overlap rules
* The chosen destination region is safe to write before or after the source region, depending on the actual layout

This is why the operator is best thought of as a carefully controlled buffer rearrangement step, not a general-purpose in-place memmove.

---

# 6. Relationship to Attention

In [`src/transformer/attention.rs`](../../../src/transformer/attention.rs), the operator is called only when `decode_only_flag` is enabled:

```rust
if decode_only_flag {
    hidden_states.lift_vector();
}
```

That means its role is tied to decode-time attention flow, not the full prefill path.

The practical implication is:

* The attention layer produces hidden states
* `LiftVector` repacks the last token vectors that matter for decode output
* Later stages can consume the compacted result more directly

---

# 7. Example From the Tests

The unit test in [`src/operators/left_vector.rs`](../../../src/operators/left_vector.rs) shows the expected behavior clearly:

* Three slices are provided
* Each slice has `last_token_flag = true`
* Each slice points to a different last-token position
* After `run()`, the copied vectors appear in the front of the buffer in slice order

That test demonstrates the core mental model:

```text
slice 0 last token -> output slot 0
slice 1 last token -> output slot 1
slice 2 last token -> output slot 2
```

---

# 8. Summary

`LiftVector` is a decode-time buffer compaction operator.

Its job is simple but important:

* Find the last token vector of each selected slice
* Copy that vector into a contiguous destination area
* Keep the result ordered and compact for downstream decode work

If you want one sentence:

> `LiftVector` turns slice-local last-token hidden states into a densely packed decode buffer by statically splitting `decode_list` across threads and copying the final vector of each eligible slice into continuous output slots.
