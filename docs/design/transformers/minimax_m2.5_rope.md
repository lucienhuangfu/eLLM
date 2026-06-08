# MiniMax M2.5 RoPE Principles

## 1. What This Document Covers

This document explains the Rotary Embedding design of `MiniMax-M2.5` and aligns it with the implementation in this repository.

It focuses on four questions:

1. How MiniMax-M2.5 RoPE relates to normal RoPE
2. Why `rope_init_fn(self.config, device)` returns both `inv_freq` and `attention_scaling`
3. What `rotary_dim = 64` means semantically
4. How the logic is implemented in this repository

---

## 2. One-Sentence Conclusion

`MiniMax-M2.5` uses **partial-dimension RoPE**.

This does not mean "the entire head dimension is rotated." Instead:

1. The first `rotary_dim = 64` dimensions of each head enter the RoPE subspace
2. Inside this subspace, adjacent pairs are still rotated using `cos/sin`
3. The remaining `64` dimensions stay as non-rotary channels

So it can be viewed as:

> `MiniMax-M2.5 = rotary subspace + standard pairwise RoPE + non-rotary residual channels`

---

## 3. Key Configuration Values

From `models/MiniMax-M2.5/config.json`, the RoPE-related fields are mainly:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `head_dim` | `128` | Total dimension of each attention head |
| `rotary_dim` | `64` | Dimensions participating in RoPE |
| `rope_theta` | `5000000` | Base frequency scale for RoPE |
| `use_qk_norm` | `true` | Q/K are normalized first |
| `qk_norm_type` | `per_layer` | Organization style of QK Norm |

The most important distinction is between `head_dim` and `rotary_dim`:

* `head_dim` is the full head width
* `rotary_dim` is the actual width of the rotary subspace

---

## 4. Core Idea of RoPE

RoPE does not add a normal position vector to the token. Instead, it writes positional information directly into the geometry of `Q/K`.

The basic idea is:

1. Treat each adjacent pair of scalar dimensions as one complex pair
2. Multiply that complex pair by a position-dependent complex rotation factor
3. Let the dot product of `Q` and `K` naturally carry relative position information

For a pair of dimensions `(x_even, x_odd)`, the rotation can be written as:

```text
x' = x_even * cos(theta) - x_odd * sin(theta)
y' = x_even * sin(theta) + x_odd * cos(theta)
```

That is the standard pairwise RoPE.

---

## 5. What `rope_init_fn` Returns

The Python code:

```python
inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
```

shows that `rope_init_fn` does not only generate a frequency table. It also returns a scale factor paired with this RoPE variant.

### 5.1 `inv_freq`

`inv_freq` describes the frequency of each rotary pair.

For the `i`-th pair:

```text
angle(pos, i) = pos * inv_freq[i]
```

Then `cos(angle)` and `sin(angle)` are generated from that angle.

### 5.2 `attention_scaling`

`attention_scaling` is a **scaling factor for the RoPE variant**, not the usual `1 / sqrt(head_dim)` in attention.

Its role is to bring the numerical scale of some RoPE variants back into a more stable range when applied.

You can think of it like this:

1. `inv_freq` decides "how to rotate"
2. `attention_scaling` decides "whether to apply an extra scale when rotating"

In Hugging Face's RoPE system, this usually appears as `attention_factor`, especially for RoPE variants with scaling strategies.

---

## 6. How MiniMax-M2.5 RoPE Works

### 6.1 First Determine the Rotary Subspace

`rotary_dim = 64` means the first 64 dimensions of each head enter RoPE.

These 64 dimensions are still split into pairs:

1. Dimensions 0 and 1 form one pair
2. Dimensions 2 and 3 form one pair
3. And so on

### 6.2 Then Generate the Frequency Table

The frequency table follows the normal RoPE logic:

1. Compute the frequency for each pair from `rotary_dim` and `rope_theta`
2. Compute the angle for each position
3. Precompute the entire `cos/sin` cache

### 6.3 Finally Apply Rotation to Q/K

At runtime, `Q` and `K` fetch the corresponding `cos/sin` values for each position and perform complex multiplication.

So:

1. The cache is prepared first
2. Runtime only does table lookup
3. Then the rotated result is written back

---

## 7. How to Understand `attention_scaling`

If we only look at the semantics of `MiniMax-M2.5`, `attention_scaling` can be understood as the matching scaling strategy for RoPE.

More specifically:

1. It is not the standard attention-logit scale
2. It is not the temperature of the softmax after the `QK` dot product
3. It is a parameter returned during RoPE initialization and paired with the frequency table

So the most accurate sentence is:

> `rope_init_fn` returns "frequency + scaling strategy," not just a plain frequency table.

If you want to align it with implementation, you should first inspect where `attention_scaling` is ultimately applied in the original Python code:

* Is it multiplied into `inv_freq`?
* Is it multiplied into the `cos/sin` cache?
* Or is it applied later to `Q/K` during RoPE application?

These three styles are semantically similar but differ in where the effect lands.

---

## 8. Relationship Between QK Norm and RoPE

`MiniMax-M2.5` also enables:

```json
"use_qk_norm": true
```

This means `Q` and `K` are normalized before entering attention.

You can think of the overall order as:

1. `Q/K` are normalized first to keep numerical stability
2. RoPE is then applied to inject positional information into phase relationships
3. The result enters the attention score computation

So:

* `QK Norm` solves scale stability
* `RoPE` solves position injection

They are chained together, not substitutes for each other.

---

## 9. Corresponding Implementation in This Repository

### 9.1 RoPE Cache Generation

In this repository, the RoPE precomputation logic is in:

* `src/transformer/rope.rs`

The core functions are:

* `inv_freqs(dim, theta)`
* `precompute_freqs_cis(dim, max_sequence_length, theta)`
* `precompute_freqs_cis_t<T>(dim, max_sequence_length, theta)`

What they do is simple:

1. Generate inverse frequencies first
2. Generate `cos/sin` values by position
3. Cache the result as a flat array

### 9.2 Attaching the Cache During Model Initialization

In `src/transformer/model.rs`, the model builds RoPE cache as `position_embedding`.

The corresponding call is:

```rust
precompute_freqs_cis_t::<f32>(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta as f32,
)
```

### 9.3 How Rotation Is Executed

The actual rotation happens mainly in:

* `src/kernel/scalar/complex_mul.rs`
* `src/kernel/x86_64/f16_512/matmul_rms_complex.rs`
* `src/operators/matmul/matmul3.rs`

`complex_mul` does standard complex multiplication:

1. Read a pair of real/imaginary parts
2. Read the corresponding `cos/sin`
3. Write back the rotated result

---

## 10. Boundaries of This Implementation

The `rope.rs` in this repository is a general cache generator. It does not itself know whether a model only uses part of the channels in `rotary_dim`.

So when integrating `MiniMax-M2.5`, two things must be clear:

1. Which channels enter the RoPE subspace
2. Where `attention_scaling` finally takes effect

In other words:

* `rotary_dim` decides "which channels are rotated"
* `attention_scaling` decides "the scaling strategy of this RoPE variant"

Do not mix these two responsibilities into one concept.

---

## 11. Minimal Mental Model

If you compress MiniMax-M2.5's RotaryEmbedding into one sentence, remember:

> For the `rotary_dim` subspace of each head, first generate the position frequency table, then apply pairwise complex rotation by position, and attach the RoPE-variant scaling factor when needed.

Shorter version:

1. Compute `inv_freq`
2. Compute `cos/sin`
3. Use those values to rotate `Q/K`
4. If the RoPE variant needs it, apply `attention_scaling`

---

## 12. Conclusion

MiniMax-M2.5's RotaryEmbedding can be understood as:

1. A variant of standard pairwise RoPE
2. Rotation only inside the `rotary_dim = 64` subspace
3. A `rope_init_fn` that returns both the frequency table and the scaling strategy

In this repository, the logic is mainly implemented by `rope.rs`, `complex_mul`, and the attention path together.
