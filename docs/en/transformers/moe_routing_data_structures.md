# MoE Routing Data Structure Changes

## 1. Old Data Structures In The Current Code

The current MoE router returns these core routing buffers:

```rust
experts_indicator // [num_experts]
indice_ptr        // [num_experts, num_tokens]
weight_ptr        // [num_experts, num_tokens]
topk_indices_ptr  // [num_tokens, num_topk]
```

### `experts_indicator`

```rust
experts_indicator[e]
```

This indicates whether expert `e` is selected by any token in the current batch or sequence chunk.

- `true`: this expert has tokens to process
- `false`: this expert has no tokens, so later expert matmul can skip it

### `indice_ptr`

```rust
indice_ptr[e * num_tokens + token]
```

Shape:

```rust
[num_experts, num_tokens]
```

Meaning:

```rust
indice_ptr[e, token] = true
```

This means token `token` is routed to expert `e`.

It is an expert-major dense bool matrix:

```text
expert 0: token0, token1, token2, ...
expert 1: token0, token1, token2, ...
expert 2: token0, token1, token2, ...
```

### `weight_ptr`

```rust
weight_ptr[e * num_tokens + token]
```

Shape:

```rust
[num_experts, num_tokens]
```

Meaning:

```rust
weight_ptr[e, token] = router_score
```

It is aligned with `indice_ptr`. The corresponding score is valid only when:

```rust
indice_ptr[e, token] == true
```

### `topk_indices_ptr`

```rust
topk_indices_ptr[token * num_topk + k]
```

Shape:

```rust
[num_tokens, num_topk]
```

Meaning:

```rust
topk_indices_ptr[token, k] = expert_id
```

This records the `k`-th top-k expert selected for token `token`.

The current layout is token-major:

```text
token 0: expert_a, expert_b, expert_c, ...
token 1: expert_d, expert_e, expert_f, ...
```

## 2. Main Problems In The Current Code

The current routing stage writes dense structures:

```rust
indice_ptr[e, token] = bool
weight_ptr[e, token] = score
```

The later gate/up and down stages then need to scan again:

```rust
for e in 0..num_experts {
    for token in 0..num_tokens {
        if indice_ptr[e, token] {
            routed_tokens.push(token)
        }
    }
}
```

This scan happens in every thread.

Main drawbacks:

1. `indice_ptr` and `weight_ptr` are dense matrices, but the number of valid route entries is only:

   ```rust
   num_tokens * topk
   ```

2. Expert matmul repeatedly scans:

   ```rust
   num_experts * num_tokens
   ```

3. Multi-threaded routing directly writes shared structures, for example:

   ```rust
   experts_indicator[e] = true
   indice_ptr[e, token] = true
   weight_ptr[e, token] = score
   ```

4. Later stages must reconstruct each expert's token list from the bool matrix.

## 3. New Global Data Structures

The proposed routing structure is an expert-major compact queue:

```rust
atomic_expert_vector // [num_experts]
index_tensor         // [num_experts, capacity_per_expert]
score_tensor         // [num_experts, capacity_per_expert]
topk_indices_ptr     // [num_tokens, num_topk]
```

For now, `capacity_per_expert` can be:

```rust
num_tokens
```

A single expert may receive every token in the worst case.

If we use the original description:

```rust
input_tensor // [topk, batch * seq]
```

it stores the same kind of information as the current `topk_indices_ptr`; only the layout differs.

I recommend keeping the current token-major layout:

```rust
topk_indices_ptr // [num_tokens, num_topk]
```

Routing produces top-k results token by token, and the down stage also uses token-contiguous access when it needs to look up the slot. This layout fits the current code better.

## 4. New Structure Details

### `atomic_expert_vector`

```rust
atomic_expert_vector[e]
```

Shape:

```rust
[num_experts]
```

Purpose:

1. Allocate global write positions for each expert during routing.
2. Store the final token count for expert `e` after routing.

Example:

```rust
base = atomic_expert_vector[e].fetch_add(local_count)
```

If a thread has `local_count` local tokens for expert `e`, it uses the atomic counter to reserve the global start position `base`.

After routing:

```rust
atomic_expert_vector[e] == token_count for expert e
```

The later expert matmul can directly read:

```rust
count = atomic_expert_vector[e]
```

### `index_tensor`

```rust
index_tensor[e * capacity_per_expert + pos]
```

Shape:

```rust
[num_experts, capacity_per_expert]
```

Meaning:

```rust
index_tensor[e, pos] = token_id
```

This stores the token id for the `pos`-th route entry of expert `e`.

Layout:

```text
expert 0: token3, token8, token19, ...
expert 1: token0, token4, ...
expert 2: token7, token9, token21, ...
```

It replaces the old:

```rust
indice_ptr[e, token] = true
```

The old structure is a bool matrix. The new structure is a compact token queue.

### `score_tensor`

```rust
score_tensor[e * capacity_per_expert + pos]
```

Shape:

```rust
[num_experts, capacity_per_expert]
```

Meaning:

```rust
score_tensor[e, pos] = router_score
```

It is aligned with `index_tensor`:

```rust
index_tensor[e, pos] = token_id
score_tensor[e, pos] = score for token_id routed to expert e
```

It replaces the old:

```rust
weight_ptr[e, token] = score
```

The old structure stores scores by original token position. The new structure stores scores by compact position inside each expert.

### `topk_indices_ptr`

```rust
topk_indices_ptr[token * num_topk + k]
```

Shape:

```rust
[num_tokens, num_topk]
```

The meaning stays the same:

```rust
topk_indices_ptr[token, k] = expert_id
```

It records the top-k expert list for each token.

If the down stage needs to know which slot expert `e` occupies for token `token`, it can look it up:

```rust
for k in 0..num_topk {
    if topk_indices_ptr[token * num_topk + k] == e {
        slot = k;
        break;
    }
}
```

If we add another structure later, we can store the slot directly and avoid this lookup. For now, we keep `topk_indices_ptr`.

## 5. New Thread-Local Mini Structures

To avoid one atomic operation for every route entry, the routing stage can use thread-local buffers:

```rust
mini_expert_vector // [num_experts]
mini_index_tensor  // [num_experts, mini_batch * topk]
mini_score_tensor  // [num_experts, mini_batch * topk]
```

### `mini_expert_vector`

```rust
mini_expert_vector[e]
```

Shape:

```rust
[num_experts]
```

Meaning:

```rust
mini_expert_vector[e] = number of route entries routed to expert e by this thread
```

It is thread-private and does not need atomics.

Example after one thread processes a token chunk:

```text
expert 0: 3
expert 1: 0
expert 2: 5
expert 3: 1
```

This means the thread has:

```text
3 route entries for expert 0
5 route entries for expert 2
1 route entry for expert 3
```

### `mini_index_tensor`

```rust
mini_index_tensor[e * mini_capacity + local_pos]
```

Shape:

```rust
[num_experts, mini_batch * topk]
```

Meaning:

```rust
mini_index_tensor[e, local_pos] = token_id
```

It temporarily stores the thread-local token list routed to expert `e`.

### `mini_score_tensor`

```rust
mini_score_tensor[e * mini_capacity + local_pos]
```

Shape:

```rust
[num_experts, mini_batch * topk]
```

Meaning:

```rust
mini_score_tensor[e, local_pos] = router_score
```

It is aligned with `mini_index_tensor`:

```rust
mini_index_tensor[e, local_pos] = token_id
mini_score_tensor[e, local_pos] = score
```

This structure is necessary; if the thread-local cache stores only token ids, the later global `score_tensor` write does not know the corresponding score.

## 6. New Routing Write Flow

### Step 1: Each thread processes its token range

The current router already splits work by token:

```rust
assign(num_tokens, thread_num, thread_id)
```

Each thread processes:

```rust
token_begin..token_end
```

### Step 2: Compute top-k inside the thread

For each token:

```text
slot 0 -> expert e0, score s0
slot 1 -> expert e1, score s1
...
```

Write the token-major top-k table:

```rust
topk_indices_ptr[token, slot] = expert_id
```

### Step 3: Bucket by expert inside the thread

For each top-k route entry:

```rust
local_pos = mini_expert_vector[expert_id]

mini_index_tensor[expert_id, local_pos] = token_id
mini_score_tensor[expert_id, local_pos] = score

mini_expert_vector[expert_id] += 1
```

This is entirely thread-local and does not need atomics.

### Step 4: Each thread reserves a global range for each expert

After a thread finishes one mini batch, for each expert:

```rust
count = mini_expert_vector[e]

if count > 0 {
    base = atomic_expert_vector[e].fetch_add(count)
}
```

`base` is the start position reserved by this thread in global `index_tensor[e, :]`.

### Step 5: Write the global compact queue

```rust
for i in 0..count {
    index_tensor[e, base + i] = mini_index_tensor[e, i]
    score_tensor[e, base + i] = mini_score_tensor[e, i]
}
```

Finally:

```rust
atomic_expert_vector[e]
```

is the total token count of expert `e`.

## 7. Old-To-New Mapping

### `experts_indicator`

Old:

```rust
experts_indicator[e]
```

New replacement:

```rust
atomic_expert_vector[e] > 0
```

### `indice_ptr`

Old:

```rust
indice_ptr[e, token] = true
```

New:

```rust
index_tensor[e, pos] = token
```

Difference:

```text
old: dense expert-token bool matrix
new: expert-major compact token queue
```

### `weight_ptr`

Old:

```rust
weight_ptr[e, token] = score
```

New:

```rust
score_tensor[e, pos] = score
```

Difference:

```text
old: score stored at the original token position
new: score stored at the compact position inside the expert queue
```

### `topk_indices_ptr`

Old:

```rust
topk_indices_ptr[token, k] = expert_id
```

Recommended new structure:

```rust
topk_indices_ptr[token, k] = expert_id
```

If we strictly follow `input_tensor` from the original description:

```rust
input_tensor[k, token] = expert_id
```

Both store the same information with different layouts.

## 8. Advantages Of The New Design

### 1. Avoid repeated bool-matrix scans in expert matmul

Old downstream code needs:

```rust
for e in 0..num_experts {
    for token in 0..num_tokens {
        if indice_ptr[e, token] {
            ...
        }
    }
}
```

New downstream code can directly use:

```rust
count = atomic_expert_vector[e]
tokens = index_tensor[e, 0..count]
scores = score_tensor[e, 0..count]
```

### 2. Better fit for expert GEMM

Each expert's tokens are stored contiguously:

```text
expert e -> token queue
```

The later gate/up/down stages can directly build expert tasks from this queue.

### 3. Less wasted dense-memory traversal

Old structure:

```rust
indice_ptr // num_experts * num_tokens
weight_ptr // num_experts * num_tokens
```

Only these route entries are actually valid:

```rust
num_tokens * topk
```

The new structure may still conservatively allocate `num_tokens` capacity per expert at first, but the access pattern only traverses the valid length:

```rust
0..atomic_expert_vector[e]
```

Later, this can be further optimized into a prefix-sum flat buffer.

### 4. Fewer multi-threaded write conflicts

Threads first write locally:

```rust
mini_expert_vector
mini_index_tensor
mini_score_tensor
```

Then each thread performs only one atomic operation per expert:

```rust
fetch_add(count)
```

For large writes, each thread writes its own reserved global range:

```rust
index_tensor[e, base..base+count]
score_tensor[e, base..base+count]
```

No mutex is needed.

## 9. Notes And Pitfalls

### 1. `index_tensor` must store token ids

Because the first dimension is already expert id:

```rust
index_tensor[e, pos]
```

the value should be:

```rust
token_id
```

not another expert id.

Correct meaning:

```rust
index_tensor[e, pos] = token_id
```

### 2. `score_tensor` must align with `index_tensor`

These two entries must describe the same route item:

```rust
index_tensor[e, pos]
score_tensor[e, pos]
```

### 3. `atomic_expert_vector` must be cleared before routing

Before every MoE routing pass:

```rust
atomic_expert_vector[e] = 0
```

Only after routing completes does it contain valid counts.

### 4. `mini_*` is only used during routing

`mini_expert_vector`, `mini_index_tensor`, and `mini_score_tensor` are thread-local caches for top-k / softmax routing.

The expert matmul stage does not use `mini_*`; it consumes:

```rust
atomic_expert_vector
index_tensor
score_tensor
topk_indices_ptr
```

### 5. The down stage must look up slot if slot is not stored

If we do not add a separate slot structure, the down stage needs to look up:

```rust
topk_indices_ptr[token, k]
```

to find the slot for expert `e` and token `token`.

This works, but it has extra cost.

## 10. Summary

Old design:

```text
routing stage:
  write dense routing table

expert matmul stage:
  every thread scans the dense routing table
  temporarily rebuilds expert -> token lists
```

New design:

```text
routing stage:
  each thread buckets routes locally
  uses atomics to reserve global positions
  directly produces expert -> token compact queues

expert matmul stage:
  directly reads each expert's token queue
  no longer scans a dense bool matrix
```

The core change is:

```rust
indice_ptr[e, token] = bool
```

becoming:

```rust
index_tensor[e, pos] = token_id
atomic_expert_vector[e] = token_count
```
