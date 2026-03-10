# CPU 上 Attention 静态并行分配说明

---

# 1️⃣ 问题背景

本文讨论 CPU 上 attention 的静态并行组织方式，目标是：

* 静态任务分配，无运行时动态抢占
* 尽量提高 CPU 利用率
* 尽量提高同一 `kv_head` 上的 K/V 访问局部性
* 不引入额外的 head 间同步
* 让 attention 的张量组织与 GQA 结构保持一致

本文聚焦的是调度结构、张量组织和并行切分方式。就实现阶段而言：

> attention 的张量变换、任务切片接入和静态遍历框架已经具备，
> 完整的 causal attention 数值计算内核仍在补全中。

---

# 2️⃣ 参数定义

| 参数 | 含义 |
| --- | --- |
| `thread_num` | CPU 并行线程数 |
| `num_attention_heads` | Q head 数 |
| `num_key_value_heads` | KV head 数 |
| `num_key_value_groups` | GQA 分组比，等于 `num_attention_heads / num_key_value_heads` |
| `batch_size` | batch 大小 |
| `seq_len` | attention kernel 中使用的序列长度 |
| `head_dim` / `head_size` | 每个 head 的维度 |
| `row_size` | 行方向 block 粒度，当前实现为 1 |
| `col_size` | 列方向 block 粒度，当前实现为 8 |
| `sequence_chunk_size` | 前向路径中当前 chunk 的序列长度 |

说明：

* GQA 通过 `num_attention_heads / num_key_value_heads` 的映射关系表达，不额外引入独立的 `group` 张量维度。
* `batch_size` 仍然是张量语义中的 batch 维，但任务输入不是直接按 batch 维展开，而是通过外部切片结构传入。
* 在前向层里更常见的是 `head_dim` 和 `sequence_chunk_size`；在底层 attention kernel 里对应的是 `head_size` 和 `seq_len`。

---

# 3️⃣ 计算结构

## 3.1 张量组织

attention 前向路径的张量组织可以概括为：

* 输入 hidden states 先生成 Q / K / V 投影
* Q 按 `num_attention_heads` 组织
* K / V 按 `num_key_value_heads` 组织
* 输出 attention 结果后，再回到拼接后的 hidden size，并与 residual 做输出投影融合

从语义上看：

* Q 的逻辑形状是 `[sequence_chunk_size, batch_size, num_attention_heads, head_dim]`
* K / V 的逻辑形状是 `[batch_size, num_key_value_heads, sequence_chunk_size, head_dim]`

这表示整体结构遵循典型的 GQA 设计：

* 多个 Q head 共享一个 KV head
* 共享关系由 `num_key_value_groups = num_attention_heads / num_key_value_heads` 决定

## 3.2 最小调度单元

attention kernel 处理的最小外部任务单元不是单独某一行，也不是单独某一个 batch，而是一个 `SequenceSlice`。

一个 slice 会携带：

* 它属于哪个 batch
* 它对应哪段 sequence/token 区间
* 这段区间的长度

因此，最小调度单元可以理解为：

```text
一个 batch 中某段连续 token 区间上的 attention 子任务
```

在这个 slice 内部，再继续按 `kv_head`、行范围、列块去展开遍历。

## 3.3 外层执行顺序

整体执行结构更接近下面这种形式：

```text
for slice in attention_list:
    定位当前 batch 和 token 区间
    for kv_head in 0 .. kv_head_num:
        对该 kv_head 的行块与列块做静态遍历
```

这里有几点值得注意：

* 外层任务来源是 `attention_list`，不是单纯按 batch 维直接均分。
* `batch` 参与任务定位，因为每个 slice 都带有 `batch_index`。
* 遍历顺序上会先固定一个 `kv_head` 再展开其内部遍历，但这是单个 slice 内部的顺序，不表示全局线程间存在严格同步的 head 调度。

---

# 4️⃣ 静态并行方式

## 4.1 并行切分原则

attention 的静态并行切分分为两层：

* 第一层：外部调度器先把 batch 内的 token 区间组织成 `SequenceSlice`
* 第二层：attention kernel 在单个 slice 内，根据 slice 的可分配工作量选择切分方式

因此它不是“只沿 batch 切”或者“只沿整条序列切”的单层模型，而是：

* 先有 slice 级任务输入
* 再有 slice 内部的静态切分

这里需要补充一个关键判断：

* 当 slice 的行块数足以覆盖线程时，采用“按 sequence 维度切分”
* 当 slice 的行块数不足以覆盖线程时，采用“按 head 切分”

因此需要分情况讨论这两种静态切分方式。

## 4.2 情况一：slice 足够长，按 sequence 维度切分

适用条件：

* 当前 slice 的行块数足以覆盖线程数
* 行维上存在足够的可分配工作量

这时采用按 sequence 维度切分。更准确地说，是沿当前 slice 的 sequence 范围去划分连续区间，而不是简单按 `ceil(seq_len / thread_num)` 平均分配行数；具体负载估算仍按下三角工作量近似均分。

原因是：

* attention 的前面几行计算量小，后面几行计算量大
* 如果只按“行数相等”切分，线程负载会偏斜
* 按三角前缀工作量估算，可以把线程映射到 sequence 维度上的不同连续区间

因此，更准确的说法是：

* 每个线程拿到的是 sequence 维度上的一个连续区间
* 这些区间追求的是“工作量近似均衡”
* 不是“行数完全平均”
* 所有线程仍遍历自己负责范围内涉及的 `kv_head`

这一模式的优先目标是：

* 贴合 causal attention 的下三角负载特征
* 尽量让线程负载接近均衡
* 保持 `kv_head` 内部遍历结构稳定

但这条规则只适用于 slice 足够长、行块数量足以支撑线程展开的情况。否则会出现：

* 某些线程根本分不到有效行区间
* 单个短 slice 无法覆盖所有核

## 4.3 情况二：slice 较短，按 head 切分

适用条件：

* 当前 slice 的行块数不足以覆盖线程数
* 继续按 sequence 维度切分会导致部分线程拿不到有效区间

这时不再沿长度方向继续细分，而是切换到 head 维切分：

* 以 `attention_head_num` 作为切分总量
* 把连续的 head 区间尽量平均地分给各线程
* 每个线程对自己负责的 head 段处理完整的 slice 行范围

需要注意的是，情况二里每个线程内部的遍历方式与情况一不同：

* 情况一中，线程先拿到 sequence 维度上的一段连续区间，再在该区间内展开遍历
* 情况二中，线程先拿到一段 attention head，再在对应 head 的整个下三角区域内做遍历
* 在这个下三角区域内，更合适的理解是按列块推进，而不是先按行块切出线程私有区域

这一模式的优先目标是：

* 提高短 slice 场景下的核覆盖率
* 避免因为 slice 太短导致大量线程空转
* 在不引入额外同步的前提下继续保持静态分配

从遍历语义上看，这时更接近下面这种形式：

```text
for slice in attention_list:
    for attention_head in assigned_head_range(thread_id):
        映射到对应的 kv_head
    在该 head 对应的下三角区域内按列块遍历
```

## 4.4 两种方式的关系

这两种切分方式不是互相叠加，而是二选一：

* 长 slice 优先按 sequence 维度切分，因为它更符合 causal attention 的负载形状
* 短 slice 优先按 head 切分，因为它更有利于把工作摊到更多核上
* 两种方式都建立在同一个 `SequenceSlice` 外层任务输入之上
* 两种方式都保持静态任务分配，不依赖运行时抢占或动态 stealing

## 4.5 Block 遍历粒度

内部遍历按 block 切分：

* 行方向粒度为 `row_size = 1`
* 列方向粒度为 `col_size = 8`

但这两个 block 参数在两种切分方式里的组织方式并不完全相同。

情况一里可以把内部遍历理解为：

* 线程先拿到 sequence 维度上的连续区间
* 再在该区间内部按 `row_chunk × col_chunk` 的二维 block 结构展开
* 当前 block 参数是 `1 × 8`

情况二里更合适的理解是：

* 线程先拿到 attention head 的连续区间
* 然后在对应 head 的完整下三角区域内按列块推进
* `row_size` 和 `col_size` 仍然定义了局部 block 粒度，但线程私有工作不再是“先切一段行块再遍历”

## 4.6 线程占用特性

“任务足够多时所有核必然沾满”并不总是成立。

更准确地说：

* 对整体批次而言，只要 slice 足够多，整体并行度通常可以做起来
* 对单个长 slice 而言，按 sequence 维度切分时仍可能因为三角切分结果为空而出现局部空转
* 对单个短 slice 而言，按 head 切分可以显著改善核覆盖率，但当 `attention_head_num` 本身也小于线程数时，仍可能存在空闲线程
* 因此单个 slice 上允许出现线程空转

也就是说，这种静态策略追求的是：

* 总体上负载可分
* 长 slice 上优先保持 causal 工作量近似均衡
* 短 slice 上优先通过 head 切分提高线程覆盖率

---

# 5️⃣ GQA 与 Causal 语义

## 5.1 GQA 语义

GQA 语义体现在：

* Q 使用 `num_attention_heads` 个 head
* K / V 使用 `num_key_value_heads` 个 head
* 每个 `kv_head` 对应一组 Q head

这种对应关系通过 head 数比例和 stride 映射隐式表达，而不是显式写成：

```text
for group in 0 .. num_key_value_groups
```

因此更合适的描述是：

* 这是一个 GQA 结构
* 组关系由 `num_attention_heads / num_key_value_heads` 隐式表示

## 5.2 Causal 语义

对 causal 语义，可以区分为两层：

* 调度层已经按“下三角工作量”来估算和切分行区间
* 逐行的 causal 访问上界还没有在内层数值 kernel 中完整实现

因此更合适的表述是：

* 已经按 causal attention 的负载特征来设计静态切分方式
* 完整的 causal 数值约束仍待在 attention 内核中补齐

---

# 6️⃣ 总结

这套 attention 静态并行方案可以概括为：

* attention 路径具备 GQA 的张量组织方式。
* 外层任务由 `SequenceSlice` 提供 slice 级输入，而不是直接按 batch 均分。
* slice 足够长时，内部线程划分按三角工作量静态分段，而不是按行数平均分配。
* slice 较短、无法用行块覆盖所有线程时，改按 `attention_head_num` 静态切分 head，并把 head 公平分配到各核。
* 内部遍历采用二维 block 结构，当前参数为 `row_size=1`、`col_size=8`。
* 整体已经形成较完整的静态调度与遍历框架，attention 内层的完整 causal 数值计算仍待补齐。

---

# 7️⃣ 核心思想一句话

> 这套方案的核心，是用 `SequenceSlice` 提供外层任务；长 slice 用三角工作量划分线程行区间，短 slice 用 `attention_head_num` 静态切分 head；并在对应 head / `kv_head` 范围内按二维 block 遍历 attention 计算。
