# CPU Attention Static Parallel Allocation

---

# 1. Problem Background

This document discusses how CPU attention is organized with static parallelism. The goals are:

* Static task assignment with no runtime preemption
* Maximize CPU utilization as much as possible
* Improve K/V access locality on the same `kv_head`
* Avoid extra synchronization between heads
* Keep attention tensor organization aligned with the GQA structure

The focus here is on scheduling structure, tensor organization, and parallel splitting. In terms of implementation status:

> Static attention splitting, block-wise traversal, and the scalar block attention kernel are already wired up.  
> The default `AttentionTrait` implementation and the `f16` / `f32` specialization paths both land on `kernel::scalar::block_flash_attention`.

---

# 2. Parameter Definitions

| Parameter | Meaning |
| --- | --- |
| `thread_num` | Number of CPU worker threads |
| `num_attention_heads` | Number of Q heads |
| `num_key_value_heads` | Number of KV heads |
| `num_key_value_groups` | GQA group ratio, equal to `num_attention_heads / num_key_value_heads` |
| `batch_size` | Batch size |
| `seq_len` | Sequence length used inside the attention kernel |
| `head_dim` / `head_size` | Dimensionality of each head |
| `row_step` | Block granularity in the row direction, currently passed as 1 by `Tensor::attention` |
| `col_step` | Block granularity in the column direction, currently passed as 8 by `Tensor::attention` |
| `sequence_length` | Sequence length of the current chunk in the forward path |

Notes:

* GQA is expressed through the mapping `num_attention_heads / num_key_value_heads`, without introducing a separate `group` tensor dimension.
* `batch_size` is still the batch dimension in tensor semantics, but task input is not expanded directly by batch dimension; it is passed in through an external slice structure.
* In the forward layer, `head_dim` and `sequence_length` are more common; in the lower-level attention kernel, the corresponding names are `head_size` and `seq_len`.
* `decode_only_flag` is stored in the `Attention` struct, but the current attention scheduling and compute path do not branch on it.

---

# 3. Compute Structure

## 3.1 Tensor Organization

The attention forward tensor layout can be summarized as:

* Input hidden states first generate Q / K / V projections
* Q is organized by `num_attention_heads`
* K / V is organized by `num_key_value_heads`
* After attention is computed, the result is merged back into the concatenated hidden size and fused with residual through the output projection

Semantically:

* The logical shape of Q is `[sequence_length, batch_size, num_attention_heads, head_dim]`
* The logical shape of K / V is `[batch_size, num_key_value_heads, sequence_length, head_dim]`

This follows the typical GQA design:

* Multiple Q heads share one KV head
* The sharing relationship is determined by `num_key_value_groups = num_attention_heads / num_key_value_heads`

## 3.2 Smallest Scheduling Unit

The smallest external task unit handled by the attention kernel is not a single row or a single batch, but a `SequenceSlice`.

A slice carries:

* `token_start_index`
* Which batch it belongs to
* Which sequence/token range it corresponds to
* The length of that range

The current attention implementation actually reads `token_start_index`, `batch_index`, `sequence_index`, and `length` to locate Q/K/V/O pointers; `lift_index` does not participate in scheduling for this operator at present.

So the smallest scheduling unit can be understood as:

```text
an attention subtask over a continuous token range within one batch
```

Inside that slice, traversal then expands over `kv_head`, row ranges, and column blocks.

## 3.3 Outer Execution Order

The overall execution looks more like this:

```text
for slice in attention_list:
    locate the current batch and token range
    choose sequence split or head split according to slice length
```

There are several points to note:

* The outer task source is `attention_list`, not a direct even split by batch.
* `batch` participates in task location because each slice carries a `batch_index`.
* In the long-slice path, a single thread traverses all `kv_head`s and corresponding local heads within the sequence range it owns.
* In the short-slice path, the scheduler first advances in `kv_head` waves, then statically assigns the flattened attention-head slots in the current wave to threads.

---

# 4. Static Parallel Strategy

## 4.1 Parallel Splitting Principles

Static parallel splitting for attention happens in two layers:

* First layer: the external scheduler organizes token ranges inside a batch into `SequenceSlice`s
* Second layer: inside a single slice, the attention kernel chooses a splitting strategy based on how much work the slice can provide

So this is not a one-layer model that "only splits by batch" or "only splits by the whole sequence." Instead:

* Slice-level tasks are provided first
* Then the slice is split statically from within

One important condition in the current implementation is:

```text
use_head_split = slice.length > 0
    && thread_num > 0
    && ceil(slice.length / row_step) < thread_num
```

That means:

* When `ceil(slice.length / row_step) < thread_num`, head splitting is used
* Otherwise sequence-dimension splitting is used

So the two static splitting modes need to be discussed separately.

## 4.2 Case 1: Long Slice, Split by Sequence Dimension

Applicable when:

* `ceil(slice.length / row_step) >= thread_num`
* or `slice.length == 0`
* or `thread_num == 0`

In this case, sequence-dimension splitting is used. More precisely, the sequence range of the current slice is split into continuous intervals rather than simply distributing rows evenly with `ceil(seq_len / thread_num)`; the work estimate still uses a triangular workload approximation.

The reason is:

* The earlier rows of attention cost less, and later rows cost more
* If rows are split only by equal count, thread load becomes imbalanced
* By estimating triangular prefix work, threads can be mapped to different continuous intervals on the sequence axis

So the more accurate description is:

* Each thread gets a continuous interval on the sequence axis
* These intervals aim for approximate work balance
* They are not "exactly equal rows"
* All threads still traverse the `kv_head`s involved in their assigned range

Implementation-wise, the current code first computes:

```text
aligned_len = floor(slice.length / row_step) * row_step
```

Then it only sends the `aligned_len` segment to the triangular-work sequence split. Although the code constructs a tail plan `tail = (aligned_len, slice.length)`, the current `visit_blocks_for_head` only consumes `row_plan.main` and does not execute `row_plan.tail`. So under the current implementation, the actual sequence-split path only computes the aligned `main` range.

The priority here is:

* Match the lower-triangular workload shape of causal attention
* Keep thread load as balanced as possible
* Keep the traversal structure within each `kv_head` stable

But this rule only applies when the slice is long enough and has enough row blocks to spread across threads. Otherwise:

* Some threads may get no valid row interval at all
* A single short slice may not cover all cores

## 4.3 Case 2: Short Slice, Split by Head

Applicable when:

* `slice.length > 0`
* `thread_num > 0`
* `ceil(slice.length / row_step) < thread_num`

In this case, the scheduler no longer keeps splitting by length. Instead, it switches to head-dimension splitting:

* The GQA structure is used to turn short-slice execution into `kv_head` wave progression
* Each wave activates as few `kv_head`s as possible while still giving all threads work
* Within one wave, threads do not permanently own different `kv_head`s; instead they jointly finish this small batch of `kv_head`s
* Only after the current wave is done does execution proceed to the next batch of `kv_head`s

Note that in case 2, work division happens in the head dimension, but the traversal method inside each head does not switch to a different kernel. The current implementation still calls the same `visit_blocks_for_head` for each assigned attention head, which means it continues to traverse the row and column area using `row_step × col_step` block iteration.

The priorities of this mode are:

* Improve core coverage in short-slice scenarios
* Let all cores finish as few `kv_head`s as possible, so fewer heads occupy the CPU cache
* Prefer concentrating attention heads sharing the same K/V within the same wave
* Avoid wasting many threads when the slice is too short or `kv_head_num` is too small
* Keep static assignment without introducing extra synchronization

From the traversal perspective, it is closer to this:

```text
for slice in attention_list:
    active_thread_num = min(thread_num, attention_head_num)
    kv_heads_per_wave = ceil(active_thread_num / attention_heads_per_kv)
    for kv_wave in contiguous kv_head waves:
        flatten the (kv_head, local_head) slots in the current wave
        statically assign continuous slot ranges to threads by thread_id
        each thread calls the same block traversal logic for every slot it owns
```

More specifically, head split under short slices is not "each thread takes a full kv_head list and runs with it." It uses wave-based static assignment.

The first step is to decide how many `kv_head`s to activate in one wave.

* Because `attention_head_num = kv_head_num * group`
* Where `group = attention_head_num / kv_head_num`
* One `kv_head` corresponds to a group of attention heads sharing the same K/V

The goal is not to let each thread quickly grab a different `kv_head`, but to ensure:

```text
active_kv_heads_per_wave * group >= thread_num
```

while keeping `active_kv_heads_per_wave` as small as possible.

That means:

* If one `kv_head` already provides enough `group`s to cover the threads, one wave only handles 1 `kv_head`
* If one `kv_head` is not enough, activate as few `kv_head`s as possible
* These `kv_head`s must be a contiguous interval; no round-robin distribution is used

So when `group >= thread_num`:

* One wave handles only 1 `kv_head`
* All threads fall under this `kv_head`
* Each thread takes a continuous portion of local heads under that `kv_head`

When `group < thread_num`:

* Multiple `kv_head`s must be activated to cover threads as much as possible
* The number of activated `kv_head`s is the minimum needed for coverage
* Threads are statically mapped onto the expanded local-head slots of this small batch of `kv_head`s

You can think of this wave as first expanding a tiny virtual task space:

```text
(kv_head_0, local_head_0 .. group-1)
(kv_head_1, local_head_0 .. group-1)
...
```

and then statically assigning those contiguous slots to threads.

So the smallest work unit a thread receives is more accurately a continuous subrange inside the flattened slot space of the current wave; once slots are restored, each slot corresponds to one specific `(kv_head, local_head)`.

This can be written as:

```text
slot_range in current wave
slot -> (kv_head, local_head)
```

The important difference is:

* The work unit is only valid within the current wave
* A thread's continuous slot range may cross local-head boundaries or adjacent `kv_head`s
* After finishing the current wave, the thread moves to the next batch of contiguous `kv_head`s as a whole
* No thread is permanently bound to a long list of heads from start to finish

The core priorities are:

* First decide how many `kv_head`s are needed for the current wave
* Then statically assign the contiguous local heads within those few `kv_head`s to threads
* After the current wave completes, all threads move to the next wave together
* Always keep contiguous-range assignment; do not use round-robin

## 4.4 Relationship Between the Two Strategies

The two splitting modes are not stacked on top of each other; they are mutually exclusive:

* Long slices are split by sequence dimension first, because that matches the workload shape of causal attention better
* Short slices are split by head first, with `kv_head` waves, so that as few `kv_head`s as possible cover as many threads as possible
* Both strategies are built on the same outer `SequenceSlice` task input
* Both keep static task assignment and do not rely on runtime preemption or dynamic stealing

## 4.5 Block Traversal Granularity

Internal traversal is block-based:

* Row granularity is `row_step = 1`
* Column granularity is `col_step = 8`

But the way these block parameters are organized is not exactly the same in the two splitting modes.

In case 1, the internal traversal can be understood as:

* The thread first gets a continuous interval on the sequence axis
* Then it expands the interval with a two-dimensional `row_chunk × col_chunk` block structure
* The current block parameters are `1 × 8`

In case 2, a better understanding is:

* The thread first gets a continuous range inside the flattened slot space of the current wave
* Each slot corresponds to one specific `(kv_head, local_head)`, and the full aligned row range of that head is traversed with `row_step × col_step`
* `row_step` and `col_step` always define local block granularity; the short-slice path does not switch to another traversal semantic like "column blocks only"
* The unit of progression between waves is a continuous `kv_head` interval, not a thread-private long-term list of heads

## 4.6 Thread Occupancy Characteristics

"All cores are definitely occupied when there is enough work" is not always true.

More precisely:

* For the overall batch, as long as there are enough slices, the overall parallelism can usually be built up
* For a single long slice, sequence-dimension splitting may still leave some threads idle because the triangular split may produce empty work for some parts

* For a single short slice, advancing by the minimum `kv_head` wave and splitting the flattened attention-head slots inside the wave can significantly improve core coverage and K/V cache utilization
* But when `attention_head_num` itself is smaller than the thread count, some threads may still remain idle
* Therefore, thread idling is allowed on a per-slice basis

So the goal of this static strategy is:

* The workload should be splittable overall
* Long slices should keep causal work approximately balanced
* Short slices should first fill all threads with as few `kv_head`s as possible, then move on to the next batch of `kv_head`s

---

# 5. GQA and Causal Semantics

## 5.1 GQA Semantics

GQA semantics are reflected in:

* Q uses `num_attention_heads` heads
* K / V use `num_key_value_heads` heads
* Each `kv_head` corresponds to a group of Q heads

This correspondence is expressed implicitly through the ratio of head counts and stride mapping, rather than explicitly written as:

```text
for group in 0 .. num_key_value_groups
```

So the more appropriate description is:

* This is a GQA structure
* The group relationship is implied by `num_attention_heads / num_key_value_heads`
* In short-slice head splitting, the scheduling priority also follows this GQA structure: first decide how many `kv_head`s are minimally needed for the current wave, then statically assign threads across continuous local heads under those `kv_head`s

## 5.2 Causal Semantics

Causal semantics can be separated into two layers:

* The scheduling layer estimates and splits sequence ranges by lower-triangular workload
* The inner scalar kernel also applies the causal upper bound row by row

The current `block_flash_attention` does:

```text
visible_col_end = min(sequence_index + row + 1, total_col_end)
row_col_end = min(col_end, visible_col_end)
```

So the current implementation does not only reflect causal semantics at the scheduling layer; the row-level visible column range is already truncated according to causal constraints in the actual numerical computation.

---

# 6. Summary

This static attention parallelization scheme can be summarized as:

* The attention path uses GQA tensor organization.
* The outer task input comes from `SequenceSlice`, not from a direct even batch split.
* When a slice is long enough, the inner thread split uses triangular work to split row intervals statically, rather than distributing rows evenly.
* When a slice is short and cannot cover all threads with row blocks, the inner split switches to `kv_head` waves: each wave activates as few contiguous `kv_head`s as possible, then assigns the flattened attention-head slots of that wave contiguously to threads.
* Internal traversal uses a two-dimensional block structure, with current parameters `row_step=1` and `col_step=8`.
* The current numerical path already performs row-by-row causal truncation through scalar `block_flash_attention`; however, `RowVisitPlan.tail` is not yet executed in the access path.

---

# 7. Core Idea in One Sentence

> The core of this scheme is to use `SequenceSlice` to provide outer tasks; let long slices split thread row intervals by triangular workload; let short slices advance in `kv_head` waves and statically assign the flattened `(kv_head, local_head)` slots of the current wave to threads; and inside each slot continue to run the same `row_step × col_step` block causal attention computation.
