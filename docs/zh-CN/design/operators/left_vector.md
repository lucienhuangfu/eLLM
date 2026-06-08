# LiftVector：把最后一个 token 的向量抬升到紧凑 decode 缓冲区

本文说明 [`src/operators/left_vector.rs`](../../../src/operators/left_vector.rs) 中实现的 `LiftVector` 算子。

在代码里，它通过 [`Tensor::lift_vector()`](../../../src/runtime/tensor.rs) 暴露，并且通常在 attention 前向路径里、`decode_only_flag` 为 `true` 时被调用。

---

# 1. 这个算子做什么

`LiftVector` 不是一个重计算算子，而是一个内存搬运算子。

它的工作是：

* 从每个被选中的 `SequenceSlice` 里取出最后一个 token 的向量
* 把这个向量拷贝到一个紧凑的输出区域
* 让输出区域按 decode 顺序连续排列

可以把它理解成：把每个 slice 的“最后一个 token 表示”从原始张量布局里抬升出来，重新压进一个连续的 decode buffer。

这个算子是对同一块张量 buffer 做就地处理的：

* 源数据来自原始的 `hidden_states`
* 目标数据写回到同一块存储中更靠前的位置
* 复制使用 `ptr::copy_nonoverlapping`，因此源区间和目标区间必须满足非重叠语义

---

# 2. 为什么需要它

在 decode 路径里，并不是 slice 里的每个 token 都同等重要。模型通常只需要每个 slice 的最后一个 token 表示，作为后续计算的 summary token。

这个算子就是专门解决这个布局问题的：

* 调度器已经知道当前有哪些 slice
* 每个 `SequenceSlice` 都带着自己的 token 区间
* `LiftVector` 把“每个 slice 的最后一个 token”整理成一个紧凑、前置的 buffer

所以它最适合这样理解：

```text
提取每个活跃 slice 的最后一个 token 向量
并按 decode 顺序紧凑排列
```

---

# 3. 数据模型

这个算子接收的是 `decode_list: &[SequenceSlice]`。

每个 `SequenceSlice` 提供：

* `token_start_index`
* `length`
* `last_token_flag`
* `batch_index`
* `sequence_index`

其中真正直接参与 `LiftVector` 的是：

* `token_start_index`
* `length`
* `last_token_flag`

输出槽位由线程分到的 slice 区间位置决定，而不是由 `batch_index` 或 `sequence_index` 决定。

`LiftVector` 里使用的向量宽度是 `self.length`，它来自 `Tensor::lift_vector()` 里的 `self.shape[1]`。

因此如果底层张量逻辑上是：

```text
[token_count, hidden_size]
```

那么每个 token 被复制的就是 `hidden_size` 这一段向量。

---

# 4. 执行流程

## 4.1 静态线程切分

`LiftVector::run()` 和仓库里的其他算子一样，使用连续区间切分：

```rust
let Some((begin, end)) = assign(total_tokens, thread_num, thread_id) else {
    return;
};
```

这意味着：

* 总 slice 数会被切成连续区间
* 每个线程只处理自己被分到的 slice 范围
* 如果当前线程没有任务，就直接返回

这里切的是 `decode_list.len()`，不是每个向量内部的 hidden 维。

## 4.2 单个 slice 的拷贝逻辑

对线程负责的每个 slice，执行顺序是：

1. 如果 `last_token_flag` 为 `false`，直接跳过
2. 计算源 token 下标：

   ```text
   source_token_index = token_start_index + length - 1
   ```

3. 计算目标槽位：

   ```text
   destination_index = begin + offset
   ```

4. 把 `self.length` 个元素从源 token 向量复制到目标槽位

因此它的本质映射就是：

```text
source: slice 的最后一个 token
dest:   decode 输出里的紧凑槽位
```

---

# 5. 关键语义

## 5.1 `last_token_flag`

`last_token_flag` 用来判断这个 slice 是否应该参与抬升。

如果它是 `false`：

* 这个 slice 会被跳过
* 不会发生拷贝
* 该 offset 对应的目标槽位保持原样

这样可以兼容“只有某些 slice 在这一轮里贡献最终 token”的布局。

## 5.2 目标区间是紧凑打包

目标槽位用的是 `begin + offset`，其中 `offset` 是线程范围内的局部序号。

这带来两个好处：

* 每个线程写的是连续的目标区域
* 最终结果按 slice 遍历顺序保持紧凑有序

所以它是一个“紧凑压缩”操作，而不是 scatter 型写回。

## 5.3 非重叠拷贝假设

实现里用的是 `ptr::copy_nonoverlapping`。

这表示代码默认：

* 源 token 向量和目标槽位满足非重叠语义
* 目标区间可以安全地在当前布局下被写入

因此它更像一个受控的 buffer 重排步骤，而不是通用的 `memmove`。

---

# 6. 和 attention 的关系

在 [`src/transformer/attention.rs`](../../../src/transformer/attention.rs) 里，这个算子只会在 `decode_only_flag` 打开时被调用：

```rust
if decode_only_flag {
    hidden_states.lift_vector();
}
```

这说明它主要服务于 decode 阶段，而不是完整的 prefill 路径。

可以把它理解成：

* attention 层先产生 hidden states
* `LiftVector` 再把 decode 真正需要的最后 token 向量整理出来
* 后续阶段就能更直接地消费这份紧凑结果

---

# 7. 测试里的例子

[`src/operators/left_vector.rs`](../../../src/operators/left_vector.rs) 里的单元测试把这个行为展示得很清楚：

* 提供了 3 个 slice
* 每个 slice 的 `last_token_flag` 都是 `true`
* 每个 slice 都指向不同的最后 token 位置
* `run()` 结束后，这些向量会按 slice 顺序出现在 buffer 前部

这个测试对应的心智模型是：

```text
slice 0 的最后 token -> 输出槽位 0
slice 1 的最后 token -> 输出槽位 1
slice 2 的最后 token -> 输出槽位 2
```

---

# 8. 总结

`LiftVector` 是一个 decode 阶段的 buffer 紧凑化算子。

它做的事情很简单，但很关键：

* 找出每个被选中 slice 的最后一个 token 向量
* 把这个向量复制到连续的目标区域里
* 让结果保持有序、紧凑，便于后续 decode 处理

一句话概括就是：

> `LiftVector` 会把 slice 内最后一个 token 的 hidden state 通过静态切分 `decode_list` 的方式，复制到连续输出槽位中，形成紧凑的 decode buffer。
