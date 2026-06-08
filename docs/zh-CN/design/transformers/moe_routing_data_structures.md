# MoE Routing 数据结构调整说明

## 1. 当前代码中的旧数据结构

当前 MoE router 返回的核心路由信息是：

```rust
experts_indicator // [num_experts]
indice_ptr        // [num_experts, num_tokens]
weight_ptr        // [num_experts, num_tokens]
topk_indices_ptr  // [num_tokens, num_topk]
```

其中：

### `experts_indicator`

```rust
experts_indicator[e]
```

表示 expert `e` 在当前 batch/sequence chunk 中是否被任何 token 选中。

- `true`：该 expert 有 token 需要处理
- `false`：该 expert 没有 token，后续 expert matmul 可以跳过

### `indice_ptr`

```rust
indice_ptr[e * num_tokens + token]
```

shape：

```rust
[num_experts, num_tokens]
```

语义：

```rust
indice_ptr[e, token] = true
```

表示 token `token` 被路由到了 expert `e`。

它是一个 expert-major 的稠密 bool 矩阵：

```text
expert 0: token0, token1, token2, ...
expert 1: token0, token1, token2, ...
expert 2: token0, token1, token2, ...
```

### `weight_ptr`

```rust
weight_ptr[e * num_tokens + token]
```

shape：

```rust
[num_experts, num_tokens]
```

语义：

```rust
weight_ptr[e, token] = router_score
```

它和 `indice_ptr` 对齐。只有当：

```rust
indice_ptr[e, token] == true
```

时，对应的 `weight_ptr[e, token]` 才是有效的 routing score。

### `topk_indices_ptr`

```rust
topk_indices_ptr[token * num_topk + k]
```

shape：

```rust
[num_tokens, num_topk]
```

语义：

```rust
topk_indices_ptr[token, k] = expert_id
```

表示 token `token` 的第 `k` 个 topk expert 是谁。

当前 layout 是 token-major：

```text
token 0: expert_a, expert_b, expert_c, ...
token 1: expert_d, expert_e, expert_f, ...
```

## 2. 当前代码的主要问题

当前 routing 阶段写的是稠密结构：

```rust
indice_ptr[e, token] = bool
weight_ptr[e, token] = score
```

后续 gate/up 和 down 阶段需要重新扫描：

```rust
for e in 0..num_experts {
    for token in 0..num_tokens {
        if indice_ptr[e, token] {
            routed_tokens.push(token)
        }
    }
}
```

而且这个扫描在每个线程里都会发生。

主要缺点：

1. `indice_ptr` 和 `weight_ptr` 是稠密矩阵，但实际有效 route 项只有：

   ```rust
   num_tokens * topk
   ```

2. expert matmul 阶段重复扫描：

   ```rust
   num_experts * num_tokens
   ```

3. 多线程 router 阶段直接写共享结构，例如：

   ```rust
   experts_indicator[e] = true
   indice_ptr[e, token] = true
   weight_ptr[e, token] = score
   ```

4. 后续阶段需要从 bool 矩阵重新构造每个 expert 的 token 列表。

## 3. 新的全局数据结构

新的 routing 结构建议改为 expert-major compact queue：

```rust
atomic_expert_vector // [num_experts]
index_tensor         // [num_experts, capacity_per_expert]
score_tensor         // [num_experts, capacity_per_expert]
topk_indices_ptr     // [num_tokens, num_topk]
```

其中 `capacity_per_expert` 可以先取：

```rust
num_tokens
```

因为单个 expert 最多可能接收所有 token。

如果使用老板原始描述中的：

```rust
input_tensor // [topk, batch * seq]
```

它和当前 `topk_indices_ptr` 存的是同类信息，只是 layout 不同。

我建议继续保留当前 token-major layout：

```rust
topk_indices_ptr // [num_tokens, num_topk]
```

因为 routing 是逐 token 产生 topk，down 阶段反查 slot 时也是按 token 连续访问，更适合当前代码。

## 4. 新结构说明

### `atomic_expert_vector`

```rust
atomic_expert_vector[e]
```

shape：

```rust
[num_experts]
```

作用：

1. routing 阶段为每个 expert 分配全局写入位置
2. routing 完成后表示 expert `e` 的 token 数量

例如：

```rust
base = atomic_expert_vector[e].fetch_add(local_count)
```

如果线程本地有 `local_count` 个 token 要写入 expert `e`，它通过 atomic 拿到全局起始位置 `base`。

routing 完成后：

```rust
atomic_expert_vector[e] == expert e 的 token_count
```

后续 expert matmul 直接读取：

```rust
count = atomic_expert_vector[e]
```

即可知道 expert `e` 有多少 token。

### `index_tensor`

```rust
index_tensor[e * capacity_per_expert + pos]
```

shape：

```rust
[num_experts, capacity_per_expert]
```

语义：

```rust
index_tensor[e, pos] = token_id
```

它表示 expert `e` 的第 `pos` 个 route 项对应哪个 token。

排列方式：

```text
expert 0: token3, token8, token19, ...
expert 1: token0, token4, ...
expert 2: token7, token9, token21, ...
```

它替代旧的：

```rust
indice_ptr[e, token] = true
```

旧结构是 bool 矩阵，新结构是 compact token queue。

### `score_tensor`

```rust
score_tensor[e * capacity_per_expert + pos]
```

shape：

```rust
[num_experts, capacity_per_expert]
```

语义：

```rust
score_tensor[e, pos] = router_score
```

它和 `index_tensor` 对齐：

```rust
index_tensor[e, pos] = token_id
score_tensor[e, pos] = token_id 路由到 expert e 的 score
```

它替代旧的：

```rust
weight_ptr[e, token] = score
```

旧结构按 token 原始位置存 score，新结构按 expert 内 compact 位置存 score。

### `topk_indices_ptr`

```rust
topk_indices_ptr[token * num_topk + k]
```

shape：

```rust
[num_tokens, num_topk]
```

语义保持不变：

```rust
topk_indices_ptr[token, k] = expert_id
```

它主要用于记录每个 token 的 topk expert 列表。

如果 down 阶段需要知道 expert `e` 是 token `t` 的第几个 slot，可以通过它反查：

```rust
for k in 0..num_topk {
    if topk_indices_ptr[token * num_topk + k] == e {
        slot = k;
        break;
    }
}
```

如果后续允许再加结构，可以直接存 slot，避免反查。但按当前整理，先保留 `topk_indices_ptr`。

## 5. 新的线程本地 mini 数据结构

为了避免每个 route 项都做 atomic，routing 阶段增加线程本地缓存：

```rust
mini_expert_vector // [num_experts]
mini_index_tensor  // [num_experts, mini_batch * topk]
mini_score_tensor  // [num_experts, mini_batch * topk]
```

### `mini_expert_vector`

```rust
mini_expert_vector[e]
```

shape：

```rust
[num_experts]
```

语义：

```rust
mini_expert_vector[e] = 当前线程本地路由到 expert e 的 route 项数量
```

它是线程私有的，不需要 atomic。

例如某个线程处理一批 token 后得到：

```text
expert 0: 3
expert 1: 0
expert 2: 5
expert 3: 1
```

表示该线程本地有：

```text
3 个 route 项给 expert 0
5 个 route 项给 expert 2
1 个 route 项给 expert 3
```

### `mini_index_tensor`

```rust
mini_index_tensor[e * mini_capacity + local_pos]
```

shape：

```rust
[num_experts, mini_batch * topk]
```

语义：

```rust
mini_index_tensor[e, local_pos] = token_id
```

它暂存当前线程内路由到 expert `e` 的 token 列表。

### `mini_score_tensor`

```rust
mini_score_tensor[e * mini_capacity + local_pos]
```

shape：

```rust
[num_experts, mini_batch * topk]
```

语义：

```rust
mini_score_tensor[e, local_pos] = router_score
```

它和 `mini_index_tensor` 对齐：

```rust
mini_index_tensor[e, local_pos] = token_id
mini_score_tensor[e, local_pos] = score
```

这个结构是我建议补充的。否则线程本地只缓存 token id，不缓存 score，后续写 `score_tensor` 时会缺少对应 score。

## 6. 新 routing 写入流程

### Step 1：每个线程处理自己的 token 范围

当前代码 router 阶段本来就是按 token 分线程：

```rust
assign(num_tokens, thread_num, thread_id)
```

每个线程处理：

```rust
token_begin..token_end
```

### Step 2：线程内计算 topk

对每个 token 得到：

```text
slot 0 -> expert e0, score s0
slot 1 -> expert e1, score s1
...
```

写入 token-major topk 表：

```rust
topk_indices_ptr[token, slot] = expert_id
```

### Step 3：线程内按 expert 分桶

对每个 topk route 项：

```rust
local_pos = mini_expert_vector[expert_id]

mini_index_tensor[expert_id, local_pos] = token_id
mini_score_tensor[expert_id, local_pos] = score

mini_expert_vector[expert_id] += 1
```

这一步完全在线程本地完成，不需要 atomic。

### Step 4：每个线程为每个 expert 申请全局写入区间

线程处理完一个 mini batch 后，对每个 expert：

```rust
count = mini_expert_vector[e]

if count > 0 {
    base = atomic_expert_vector[e].fetch_add(count)
}
```

`base` 是该线程在全局 `index_tensor[e, :]` 中拿到的起始位置。

### Step 5：写入全局 compact queue

```rust
for i in 0..count {
    index_tensor[e, base + i] = mini_index_tensor[e, i]
    score_tensor[e, base + i] = mini_score_tensor[e, i]
}
```

最后：

```rust
atomic_expert_vector[e]
```

就是 expert `e` 的总 token 数。

## 7. 新旧结构对应关系

### `experts_indicator`

旧结构：

```rust
experts_indicator[e]
```

新结构中可以由：

```rust
atomic_expert_vector[e] > 0
```

替代。

### `indice_ptr`

旧结构：

```rust
indice_ptr[e, token] = true
```

新结构变成：

```rust
index_tensor[e, pos] = token
```

区别：

```text
旧: expert-token 稠密 bool 矩阵
新: expert-major compact token queue
```

### `weight_ptr`

旧结构：

```rust
weight_ptr[e, token] = score
```

新结构变成：

```rust
score_tensor[e, pos] = score
```

区别：

```text
旧: score 按原始 token 位置存
新: score 按 expert 内 compact 位置存
```

### `topk_indices_ptr`

旧结构：

```rust
topk_indices_ptr[token, k] = expert_id
```

新结构建议保持：

```rust
topk_indices_ptr[token, k] = expert_id
```

如果严格使用老板描述的 `input_tensor`，则是：

```rust
input_tensor[k, token] = expert_id
```

两者存的信息一样，layout 不同。

## 8. 新方案的优势

### 1. 避免 expert matmul 阶段重复扫描 bool 矩阵

旧方案后续需要：

```rust
for e in 0..num_experts {
    for token in 0..num_tokens {
        if indice_ptr[e, token] {
            ...
        }
    }
}
```

新方案直接：

```rust
count = atomic_expert_vector[e]
tokens = index_tensor[e, 0..count]
scores = score_tensor[e, 0..count]
```

### 2. 更适合 expert GEMM

每个 expert 的 token 连续存放：

```text
expert e -> token queue
```

后续 gate/up/down 可以直接按 expert 构造 task。

### 3. 减少稠密内存浪费

旧结构：

```rust
indice_ptr // num_experts * num_tokens
weight_ptr // num_experts * num_tokens
```

实际有效 route 项只有：

```rust
num_tokens * topk
```

新结构虽然每个 expert 仍可先保守分配 `num_tokens` 容量，但访问模式只遍历有效长度：

```rust
0..atomic_expert_vector[e]
```

后续还可以进一步优化成 prefix-sum flat buffer。

### 4. 多线程写入冲突更少

线程先写本地：

```rust
mini_expert_vector
mini_index_tensor
mini_score_tensor
```

然后每个 expert 只做一次 atomic：

```rust
fetch_add(count)
```

大量数据写入时，每个线程写自己独占的全局区间：

```rust
index_tensor[e, base..base+count]
score_tensor[e, base..base+count]
```

不需要 mutex 锁。

## 9. 需要注意的问题

### 1. `index_tensor` 必须存 token id

因为第一维已经是 expert id：

```rust
index_tensor[e, pos]
```

所以里面应该存：

```rust
token_id
```

而不是再存 expert id。

正确语义：

```rust
index_tensor[e, pos] = token_id
```

### 2. `score_tensor` 要和 `index_tensor` 对齐

必须保证：

```rust
index_tensor[e, pos]
score_tensor[e, pos]
```

描述的是同一个 route 项。

### 3. `atomic_expert_vector` 在 routing 前要清零

每次 MoE routing 开始前：

```rust
atomic_expert_vector[e] = 0
```

routing 完成后它才是有效 count。

### 4. `mini_*` 只用于 routing 阶段

`mini_expert_vector`、`mini_index_tensor`、`mini_score_tensor` 是 topk / softmax routing 阶段的线程本地缓存。

expert matmul 阶段不使用 `mini_*`，而是消费：

```rust
atomic_expert_vector
index_tensor
score_tensor
topk_indices_ptr
```

### 5. down 阶段如果不存 slot，需要反查

如果不新增 slot 结构，down 阶段需要通过：

```rust
topk_indices_ptr[token, k]
```

反查 expert `e` 对应 token `token` 的 slot。

这个可以工作，但会有额外开销。

## 10. 总结

旧方案：

```text
router 阶段:
  写 dense routing table

expert matmul 阶段:
  每个线程扫描 dense routing table
  临时重建 expert -> token 列表
```

新方案：

```text
router 阶段:
  每个线程本地 mini 分桶
  用 atomic 分配全局位置
  直接生成 expert -> token compact queue

expert matmul 阶段:
  直接读取 expert 的 token queue
  不再扫描 dense bool 矩阵
```

核心变化是：

```rust
indice_ptr[e, token] = bool
```

变成：

```rust
index_tensor[e, pos] = token_id
atomic_expert_vector[e] = token_count
```
