# High-Performance Matrix Multiplication Principles

This document uses the `MatMul`, `MatMulAdd`, and `MatMulSigmoid` implementations in this repository to explain why high-performance matrix multiplication on CPU needs blocking, why `B` needs to be packed, why thread-private scratch buffers are needed, and what problem each of these designs solves.

---

# 1. Computation Goal and Data Layout

Standard matrix multiplication can be written as:

```text
C = A × B
```

Where:

* `A` has shape `M × K`
* `B` has shape `K × N`
* `C` has shape `M × N`

In this repository, `B` is often not used directly in the traditional `K × N` layout. Instead, it is first passed in as `B_nt[N × K]`, meaning a row-major `N × K` layout. The reason is:

* Upper-layer weights are often already stored in a format close to "one row per output channel"
* The kernel can pack `B` more easily into `Kc × NR`
* There is no need to repeatedly perform a full transpose during construction

This is consistent across `MatMul`, `MatMulAdd`, and `MatMulSigmoid`.

---

# 2. Why Naive Multiplication Is Slow

A naive implementation usually uses three nested loops:

```text
for i in 0..M
    for j in 0..N
        for k in 0..K
            C[i, j] += A[i, k] * B[k, j]
```

Its main problems are:

1. The access pattern of `B[k, j]` is cache-unfriendly and leads to many cache misses
2. It does only a small amount of computation per memory access, so arithmetic intensity is too low
3. It does not exploit SIMD or register reuse
4. In multithreading, splitting by element or row too finely increases scheduling and synchronization overhead

The core of high-performance matrix multiplication is solving these four problems one by one.

---

# 3. Key Terms and Parameters

## 3.1 Common Blocking Terms

The following symbols are commonly used in matrix multiplication:

* `MB`: macro-block height for `A` and `C`
* `NB`: macro-block width for `B` and `C`
* `KC`: block length in the `K` direction
* `MR`: number of rows processed by the micro-kernel at once
* `NR`: number of columns processed by the micro-kernel at once

Notes:

* `MR` / `NR` are common terms in GEMM and micro-kernel contexts, but they are not the only naming convention
* Different projects may use different letter combinations for the same size concepts
* In this document and this repository, we explicitly define them as "rows / columns processed by the micro-kernel at once"

## 3.2 `MatMulParams`

This repository folds these size parameters into `MatMulParams`:

```rust
pub struct MatMulParams {
    pub a_row_step_macro: usize, // MB / lda semantics reuse
    pub b_row_step_macro: usize, // NB / ldc semantics reuse
    pub column_step_macro: usize, // KC
    pub a_row_step_micro: usize, // MR
    pub b_row_step_micro: usize, // NR
}
```

The important part is not the field names themselves, but the two layers of meaning they express:

* Macro blocks: how the outer layer is split
* Micro blocks: how large each micro-kernel is

## 3.3 Semantic Conventions Used Here

For this document, you can understand the fields as:

* `a_row_step_macro` approximately corresponds to `MB` or `lda`
* `b_row_step_macro` approximately corresponds to `NB` or `ldc`
* `column_step_macro` corresponds to `KC`
* `a_row_step_micro` corresponds to `MR`
* `b_row_step_micro` corresponds to `NR`

Different kernels may map these fields to strides differently, but the core idea remains the same:

* Macro blocks decide outer splitting
* Micro blocks decide micro-kernel size
* `KC` determines how long a `K` panel is packed at a time

---

# 4. Main Chain of Single-Thread High-Performance GEMM

## 4.1 Blocking

Blocking, also called tiling, is the most important principle in high-performance matrix multiplication.

After splitting a large matrix into smaller blocks:

* A small `C` tile is easier to keep in registers or L1
* A small block of `A` can be reused multiple times
* A small panel of `B` can be packed once and fed repeatedly to multiple `A` sub-blocks

## 4.2 Packing `B`

In high-performance GEMM, `B` is often not computed directly from its original layout. Instead, it is first packed into a continuous panel of size `Kc × NR`.

The reason is straightforward:

* The micro-kernel wants to read contiguous memory
* Contiguous access is easier for cache prefetching
* SIMD load/store is more efficient
* The access order of `B` then matches the computation order

In this repository, the `MatMul` family prepares a `KC × NR` `b_panel_pool` for each thread, and then copies `B_nt` into that thread-private panel before computation.

The point of this step is:

* Turn "scattered reads" into "sequential reads"
* Turn "column jumps" into a continuous panel
* Pay the cost of repeated access only once up front

One thing to note: although `B_nt` itself is also row-major continuous data, its storage layout is not the same as the computation layout required by the micro-kernel.

* `B_nt[N × K]` describes how the raw weights are stored
* `Kc × NR` describes how the micro-kernel consumes the data

So packing is not just about making non-contiguous data contiguous. It is about rearranging row-major `B_nt` into a panel form that is more suitable for micro-kernel consumption.

## 4.3 Register Blocking and SIMD

A real micro-kernel usually does not compute the whole large block at once. Instead, it computes a very small output tile, for example:

* `MR = 3`
* `NR = 32`

The reason is:

* The `MR × NR` sub-block of `C` can stay in registers as much as possible
* One scalar from `A` can be broadcast and reused in FMA operations
* One vector from `B` can be shared across multiple `A` rows

In `src/kernel/x86_64/f16_512/matmul_block.rs`, `3 × 32` is a typical broadcast-style micro-kernel:

* `A` takes 3 rows
* `B_panel` takes 32 columns
* It performs broadcast multiply-add along `Kc`

The essence of this design is to increase how much computation you get from each unit of memory read.

Modern CPUs are good at matrix multiplication because they have:

* SIMD vector instructions
* FMA instructions
* Multi-level caches

The benefit of FMA is:

* One instruction performs multiplication and addition together
* Intermediate results do not need to be written out separately
* Floating-point throughput is higher

In the FP16 path, `_mm512_fmadd_ph` is a typical example.

## 4.4 Thread-Private Scratch

If multiple threads share the same packed buffer, you get:

* Contention
* False sharing
* Unnecessary synchronization

So in this repository, each thread has its own `b_panel_pool`, and `MatMulSigmoid` additionally prepares an `acc_pool`.

The benefits are:

* No write conflicts between threads
* Clear scratch lifetime
* Stable access patterns
* Easier to keep high throughput

## 4.5 Static Splitting

This design uses static task assignment instead of dynamic work stealing.

The reason is:

* Matrix multiplication tile workloads are predictable
* Static splitting has no runtime scheduling cost
* It is more cache-friendly

In `MatMul::run`, the overall tiles are first split across threads by `assign(total_tiles, thread_num, thread_id)`, and then each thread advances its own internal `K` loop and packing flow.

---

# 5. Multithreaded Tiling and Continuity

## 5.1 Splitting the Output Matrix by Tile

If you want to make matrix multiplication multithreaded, the most common and safest method is to split the output matrix by tile rather than by input element.

The recommended flow is:

```text
1. Split C into MB × NB output tiles
2. Treat each tile as an independent task
3. Use static assignment to divide task ranges among threads
4. Inside each thread, do KC blocking along K
5. Each thread uses its own B_panel / accumulator scratch
```

This gives:

* Each tile writes to an independent area of `C`, so there is no write conflict
* Each thread can keep reusing its own scratch instead of reallocating constantly
* Tile boundaries are clear, which helps static splitting and load control
* Threads rarely need synchronization with each other

## 5.2 Why Continuity Matters

If you want multithreaded splitting to be not only independent but also continuous, the most important principle is: **make the thread's tasks align with the row-major layout of `C`**.

For a row-major output matrix like in this repository, a good thread-task organization is:

* First split `C` into `MB × NB` tiles
* Flatten the `tiles_m × tiles_n` grid into a one-dimensional task sequence
* Let `assign()` distribute continuous task ranges
* Try to make the same thread process adjacent `tm` / `tn` ranges

The direct benefits are:

* One thread writes back contiguous `C` regions
* L1 / L2 prefetch is more likely to hit
* Threads are less likely to interleave writes on the same cache line
* The tile access pattern becomes more stable and easier for the CPU to predict

In practice, there are usually two preferences:

1. **Prefer a continuous `m` band**
   * Let one thread process a continuous row band as much as possible
   * Best for `C` write-back locality
   * Suitable when the output matrix is wide and each row has a large workload

2. **Advance sequentially over `n` inside the continuous `m` band**
   * Let the thread scan through that row band in column-block order
   * Preserve tile spatial locality
   * Suitable when `NB` is already aligned and `n`-direction tiles are regular enough

What you want to avoid is round-robin distribution, where tasks are interleaved across threads. For example:

* `thread 0` gets `0, 4, 8, ...`
* `thread 1` gets `1, 5, 9, ...`

This may look "balanced," but it forces each thread to jump around, which is usually bad for cache locality.

So the safer strategy is:

* **Continuous task ranges**
* **Continuous traversal order within a thread**
* **Continuous write-back regions**

If the number of tiles itself is too small, even a continuous split may not fully occupy all threads. In that case, you usually need more parallelism from a higher layer, such as:

* Smaller tile sizes
* More batch-level parallelism
* Scheduling different slices or different samples together at a higher level

## 5.3 The Role of `assign()`

`assign()` in this repository is closer to "continuous range splitting" than to "rebalancing by prefix sum."

That means its focus is:

* Simplicity
* Low overhead
* Preserving continuity

For regular GEMM, this is usually enough. If tasks are highly imbalanced, a prefix-style scheduler becomes more useful.

## 5.4 How Tile Access Proceeds After `assign()`

`assign(tiles, thread_num, thread_id)` returns a continuous range `[tb, te)`, and the thread processes tile IDs in that range one by one.

In `MatMul::run()`, each tile ID is first converted back into 2D coordinates:

```text
tm = t / tiles_n
tn = t % tiles_n
m0 = tm * mb
n0 = tn * nb
```

So the thread access order can be understood as:

1. Get a continuous range of tile IDs
2. Map each tile ID back to `(tm, tn)`
3. Use `(m0, n0)` as the top-left corner of the current tile
4. Continue to expand inside the tile along `K`, `N`, and `M`

More specifically, the internal order of a tile is usually:

```text
for k0 in 0..K step KC
    pack the B_panel for the current tile
    for nt in 0..n_blk step NR
        for mi in 0..m_blk step MR
            run the micro-kernel for one MR × NR sub-block
```

This means:

* `assign()` decides which threads handle which tiles
* The tile's internal loop decides how the thread finishes that tile

For row-major `C`, continuous `tn` values make a thread access adjacent column tiles under the same `tm` first, so write-back is usually continuous as well.

For `MatMulSigmoid`, the order is the same except for an additional `acc_ptr` layer in the middle:

* First accumulate using `b_panel_ptr`
* Then write the result to `acc_ptr`
* Finally apply `bias` and `sigmoid` together

So from the code execution point of view, the post-`assign()` tile traversal can be summarized as:

> The thread first takes continuous tiles in order, then maps the tile back to a 2D position, and then continues expanding inside the tile along the `KC × NR × MR` hierarchy.

---

# 6. Execution Chain in the Code

## 6.1 `MatMul`

The main path of `MatMul` can be summarized as:

```text
new()
  -> preallocate thread-private b_panel_pool
  -> run()
      -> compute tile range
      -> static split with assign()
      -> pack B panel
      -> call micro-kernel
```

The corresponding file is `src/operators/matmul/matmul.rs`.

Two key points:

* Before entering `run()`, each thread already has its own scratch buffer
* The compute stage no longer allocates; it only packs and computes

## 6.2 `MatMulAdd`

`MatMulAdd` does not mean plain `C = A × B`; instead it means:

```text
C = residual + A × B
```

So it first copies `residual` into `output`, then continues to accumulate matmul on top of it.

This has two benefits:

* The final output is written only once to the target matrix
* Fusing residual and matmul reduces extra intermediate buffers

The corresponding file is `src/operators/matmul/matmul_add.rs`.

## 6.3 `MatMulSigmoid`

`MatMulSigmoid` adds one more layer of accumulator management on top of the normal GEMM chain.

Its compute semantics can be understood as:

```text
acc = A × B + bias
output = sigmoid(acc)
```

`acc_pool` is the thread-private buffer for this intermediate accumulation.

Its value is:

* The `A × B` accumulation can land safely in a continuous scratch first
* Sigmoid is applied only at the end, avoiding state changes while computing
* For fused kernels, the accumulator structure is clearer and easier to combine with different bias strategies

`acc_pool` is preallocated as `threads × (MB × NB)`, meaning:

* One independent accumulator area per thread
* Each area is large enough to hold the intermediate result of one tile

The corresponding code is in `src/operators/matmul/matmul_sigmoid.rs`.

One thing to note: `b_panel_pool` and `acc_pool` solve two different problems.

* `b_panel_pool` packs `B_nt` into a continuous panel suitable for the micro-kernel
* `acc_pool` stores intermediate accumulation results for the current tile

The former serves "input access efficiency," while the latter serves "intermediate computation state management."

## 6.4 `MatMulSigmoid::run()`

The approximate order of `MatMulSigmoid::run()` is:

```text
1. Compute m_run from prefill_size
2. Pad M up to a multiple of MR
3. Compute tiles_m / tiles_n
4. Use assign() to divide tiles among threads
5. Each thread takes its own b_panel_ptr and acc_ptr
6. Call block_matmul_sigmoid::matmul_sigmoid for each tile
```

This shows that `acc_pool` is not a globally shared result cache. It is a thread-local, tile-reusable intermediate workspace.

## 6.5 Kernel Layer

The real block computation logic is not in the operator layer, but in the kernel layer:

* `src/kernel/scalar/matmul_block.rs`
* `src/kernel/scalar/block_matmul_sigmoid.rs`
* `src/kernel/x86_64/f16_512/matmul_block.rs`

They are responsible for:

* `matmul_block`: the most basic block GEMM micro-kernel
* `block_matmul_sigmoid`: apply sigmoid after block multiplication
* `f16_512::matmul_block`: the 3 × 32 micro-kernel under the FP16 AVX-512 path

In other words, the operator layer is responsible for "task organization," while the kernel layer is responsible for "how the math is actually done."

---

# 7. What Can Still Be Improved

Based on this document, the following directions are still worth optimizing:

## 7.1 Pre-pack Static Weights

If `B` is a model weight and is static during inference, the ideal approach is:

* Pre-pack `B_nt` into a fixed panel layout during layer initialization
* Consume the packed `B` directly for every GEMM later
* Avoid repacking on every tile at runtime

This is usually one of the easiest places to gain performance.

## 7.2 Stronger Register Blocking

There is already an `MR × NR` micro-kernel idea. The next step is to reduce dependency chains further:

* Keep more accumulators alive at once
* Interleave `load A`, `load B`, and FMA
* Give the CPU more independent instructions for out-of-order execution

## 7.3 Make Parameters More Cache-Friendly

`MB / NB / KC / MR / NR` are not fixed truths; they should be tuned to the actual hardware:

* If `KC` is too large, pressure on `B_panel` increases
* If `KC` is too small, packing and loop overhead increase
* If `MB` or `NB` are not chosen well, tiles become too fragmented or too large

## 7.4 Double Buffering and Prefetch

If both packing and compute are significant, consider:

* One buffer computing
* Another buffer prefetching or packing the next block

Software prefetch can also help, but it is usually better to validate the actual gain first before putting it into the hot path.

## 7.5 Higher-Level Parallelism

When `tiles_m × tiles_n` is not enough, you can also bring higher-level batch, slice, or sample dimensions into scheduling so that all threads are truly kept busy.

---

# 8. A Simplified Mental Model

If you want to compress this implementation into a mental model, it looks like this:

```text
MatMul / MatMulAdd / MatMulSigmoid
    -> static tile splitting
    -> per-thread private scratch
        -> b_panel_pool: continuous panel of B
        -> acc_pool: intermediate accumulation result
    -> block / micro-kernel
        -> MR × NR register blocking
        -> SIMD / FMA
    -> write back output
```

The key idea is:

* The upper layer solves "how to split the work"
* The middle layer solves "how to store the data"
* The lower layer solves "how to compute each small block quickly"

If all three layers are done well, matrix multiplication performance usually improves noticeably.
