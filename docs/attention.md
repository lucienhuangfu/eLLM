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

> attention 的静态切分、按块遍历和 scalar block attention 内核已经接通，
> 当前 `AttentionTrait` 的默认实现以及 `f16` / `f32` 特化路径都落到 `kernel::scalar::block_flash_attention`。

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
| `row_step` | 行方向 block 粒度，当前 `Tensor::attention` 调用传入为 1 |
| `col_step` | 列方向 block 粒度，当前 `Tensor::attention` 调用传入为 8 |
| `sequence_chunk_size` | 前向路径中当前 chunk 的序列长度 |

说明：

* GQA 通过 `num_attention_heads / num_key_value_heads` 的映射关系表达，不额外引入独立的 `group` 张量维度。
* `batch_size` 仍然是张量语义中的 batch 维，但任务输入不是直接按 batch 维展开，而是通过外部切片结构传入。
* 在前向层里更常见的是 `head_dim` 和 `sequence_chunk_size`；在底层 attention kernel 里对应的是 `head_size` 和 `seq_len`。
* `decode_only_flag` 会被保存在 `Attention` 结构体中，但当前 attention 调度与计算路径并未使用它参与分支判断。

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

* `token_start_index`
* 它属于哪个 batch
* 它对应哪段 sequence/token 区间
* 这段区间的长度

当前 attention 实现实际会读取 `token_start_index`、`batch_index`、`sequence_index` 和 `length` 来定位 Q/K/V/O 指针；`lift_index` 目前不参与该算子的调度。

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
    根据 slice 长度选择 sequence split 或 head split
```

这里有几点值得注意：

* 外层任务来源是 `attention_list`，不是单纯按 batch 维直接均分。
* `batch` 参与任务定位，因为每个 slice 都带有 `batch_index`。
* 长 slice 路径中，单个线程会遍历它负责的 sequence 区间内的全部 `kv_head` 和对应 local head。
* 短 slice 路径中，会先按 `kv_head` 波次推进，再把当前波次展平后的 attention-head 槽位静态分给线程。

---

# 4️⃣ 静态并行方式

## 4.1 并行切分原则

attention 的静态并行切分分为两层：

* 第一层：外部调度器先把 batch 内的 token 区间组织成 `SequenceSlice`
* 第二层：attention kernel 在单个 slice 内，根据 slice 的可分配工作量选择切分方式

因此它不是“只沿 batch 切”或者“只沿整条序列切”的单层模型，而是：

* 先有 slice 级任务输入
* 再有 slice 内部的静态切分

这里需要补充一个关键判断。当前实现中的分支条件是：

```text
use_head_split = slice.length > 0
    && thread_num > 0
    && ceil(slice.length / row_step) < thread_num
```

也就是说：

* 当 `ceil(slice.length / row_step) < thread_num` 时，采用“按 head 切分”
* 否则采用“按 sequence 维度切分”

因此需要分情况讨论这两种静态切分方式。

## 4.2 情况一：slice 足够长，按 sequence 维度切分

适用条件：

* `ceil(slice.length / row_step) >= thread_num`
* 或者 `slice.length == 0`
* 或者 `thread_num == 0`

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

实现细节上，当前代码会先计算：

```text
aligned_len = floor(slice.length / row_step) * row_step
```

然后只对 `aligned_len` 这一段调用按三角工作量的 sequence split。代码里虽然构造了 `tail = (aligned_len, slice.length)` 这一尾段计划，但当前 `visit_blocks_for_head` 只消费 `row_plan.main`，并不会执行 `row_plan.tail`。因此按现状实现，sequence split 路径真正参与计算的是 `main` 对应的对齐区间。

这一模式的优先目标是：

* 贴合 causal attention 的下三角负载特征
* 尽量让线程负载接近均衡
* 保持 `kv_head` 内部遍历结构稳定

但这条规则只适用于 slice 足够长、行块数量足以支撑线程展开的情况。否则会出现：

* 某些线程根本分不到有效行区间
* 单个短 slice 无法覆盖所有核

## 4.3 情况二：slice 较短，按 head 切分

适用条件：

* `slice.length > 0`
* `thread_num > 0`
* `ceil(slice.length / row_step) < thread_num`

这时不再沿长度方向继续细分，而是切换到 head 维切分：

* 先利用 GQA 结构，把短 slice 的执行改成按 `kv_head` 波次推进
* 每一波只激活尽量少的 `kv_head`，但要尽量让全部线程都有活做
* 当前这一波内，线程不是各自长期独占不同 `kv_head`，而是共同完成这一小批 `kv_head`
* 只有这一波完成后，才继续下一批 `kv_head`

需要注意的是，情况二里线程分工发生在 head 维，但单个 head 内部的遍历方式并没有换成另一套内核。当前实现仍然会对被分配到的每个 attention head 调用同一个 `visit_blocks_for_head`，也就是继续按 `row_step × col_step` 的二维块遍历其行列区域。

这一模式的优先目标是：

* 提高短 slice 场景下的核覆盖率
* 让所有核先完成尽量少的 `kv_head`，从而让尽量少的 head 占用全部 CPU cache
* 优先保持共享同一 K/V 的 attention head 在同一波次内被集中处理
* 避免因为 slice 太短或 `kv_head_num` 过少导致大量线程空转
* 在不引入额外同步的前提下继续保持静态分配

从遍历语义上看，这时更接近下面这种形式：

```text
for slice in attention_list:
    active_thread_num = min(thread_num, attention_head_num)
    kv_heads_per_wave = ceil(active_thread_num / attention_heads_per_kv)
    for kv_wave in contiguous kv_head waves:
        展平当前波次内的 (kv_head, local_head) 槽位
        根据 thread_id 把连续槽位区间静态分给线程
        线程对自己拿到的每个槽位调用同一套 block 遍历逻辑
```

更具体地说，短 slice 下的 head split 不是“线程各自拿完整的 kv_head 列表一路做完”，而是采用按波次推进的静态分配。

第一步先决定一波里同时激活多少个 `kv_head`。

* 因为 `attention_head_num = kv_head_num * group`
* 其中 `group = attention_head_num / kv_head_num`
* 一个 `kv_head` 对应一组共享同一 K/V 的 attention head

目标不是让每个线程尽快拿到不同的 `kv_head`，而是让：

```text
active_kv_heads_per_wave * group >= thread_num
```

同时 `active_kv_heads_per_wave` 要尽可能小。

这意味着：

* 如果单个 `kv_head` 下的 `group` 已经足以覆盖线程，那么一波只做 1 个 `kv_head`
* 如果单个 `kv_head` 不足以覆盖线程，就同时激活尽量少的多个 `kv_head`
* 这些 `kv_head` 必须是连续区间，不做轮转式分发

因此当 `group >= thread_num` 时：

* 一波只处理 1 个 `kv_head`
* 全部线程都落在这个 `kv_head` 下
* 每个线程拿该 `kv_head` 下连续的一段 local head

当 `group < thread_num` 时：

* 需要同时激活多个 `kv_head` 才能把线程尽量铺满
* 激活的 `kv_head` 数是满足覆盖线程所需的最小值
* 线程会被静态映射到这一小批 `kv_head` 展开的 local head 槽位上

可以把这一波看成先展开一个很小的虚拟任务空间：

```text
(kv_head_0, local_head_0 .. group-1)
(kv_head_1, local_head_0 .. group-1)
...
```

然后再把这些连续槽位静态分给线程。

因此线程拿到的最小工作单元更准确地说是当前波次展平槽位空间中的一个连续子区间；把槽位还原后，每个槽位对应一个具体的 `(kv_head, local_head)`。

可以写成：

```text
slot_range in current wave
slot -> (kv_head, local_head)
```

但这里的关键区别是：

* 这个工作单元只在当前波次内有效
* 一个线程拿到的连续槽位区间可能跨越 local head 边界，也可能跨越相邻 `kv_head`
* 线程完成当前波次后，会整体推进到下一批连续 `kv_head`
* 不是某个线程从头到尾长期绑定一串 `kv_head`

这一规则的核心优先级是：

* 先决定当前波次里最少需要多少个 `kv_head`
* 再在这少量 `kv_head` 内把 local head 连续静态分给线程
* 当前波次完成后，所有线程再一起推进到下一波
* 始终保持连续区间分配，不做轮转式分配

## 4.4 两种方式的关系

这两种切分方式不是互相叠加，而是二选一：

* 长 slice 优先按 sequence 维度切分，因为它更符合 causal attention 的负载形状
* 短 slice 优先按 head 切分，并按 `kv_head` 波次推进，优先让更少的 `kv_head` 覆盖更多线程
* 两种方式都建立在同一个 `SequenceSlice` 外层任务输入之上
* 两种方式都保持静态任务分配，不依赖运行时抢占或动态 stealing

## 4.5 Block 遍历粒度

内部遍历按 block 切分：

* 行方向粒度为 `row_step = 1`
* 列方向粒度为 `col_step = 8`

但这两个 block 参数在两种切分方式里的组织方式并不完全相同。

情况一里可以把内部遍历理解为：

* 线程先拿到 sequence 维度上的连续区间
* 再在该区间内部按 `row_chunk × col_chunk` 的二维 block 结构展开
* 当前 block 参数是 `1 × 8`

情况二里更合适的理解是：

* 线程先拿到当前波次展平槽位空间中的一段连续区间
* 每个槽位对应一个具体的 `(kv_head, local_head)`，并对这个 head 的完整对齐行区间做 `row_step × col_step` 遍历
* `row_step` 和 `col_step` 始终定义局部 block 粒度，短 slice 路径并没有切换成“只按列块、不按行块”的另一套遍历语义
* 波次之间的推进单位是连续 `kv_head` 区间，而不是线程私有的长期 head 列表

## 4.6 线程占用特性

“任务足够多时所有核必然沾满”并不总是成立。

更准确地说：

* 对整体批次而言，只要 slice 足够多，整体并行度通常可以做起来
* 对单个长 slice 而言，按 sequence 维度切分时仍可能因为三角切分结果为空而出现局部空转
* 对单个短 slice 而言，按最小 `kv_head` 波次推进，并在波次内部拆分展平后的 attention-head 槽位，可以显著改善核覆盖率与 K/V cache 利用率
* 但当 `attention_head_num` 本身也小于线程数时，仍可能存在空闲线程
* 因此单个 slice 上允许出现线程空转

也就是说，这种静态策略追求的是：

* 总体上负载可分
* 长 slice 上优先保持 causal 工作量近似均衡
* 短 slice 上优先让尽量少的 `kv_head` 占满全部线程，再推进到下一批 `kv_head`

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
* 在短 slice 的 head split 中，调度优先级也遵循这一 GQA 结构：先决定当前波次最少需要多少个 `kv_head`，再在这些 `kv_head` 下按连续 local head 静态分配线程

## 5.2 Causal 语义

对 causal 语义，可以区分为两层：

* 调度层按“下三角工作量”来估算和切分 sequence 区间
* 内层 scalar kernel 也已经逐行施加 causal 上界

当前 `block_flash_attention` 的做法是：

```text
visible_col_end = min(sequence_index + row + 1, total_col_end)
row_col_end = min(col_end, visible_col_end)
```

因此当前实现并不是只有调度层体现 causal 语义；在实际数值计算时，行级可见列范围也已经按 causal 约束截断。

---

# 6️⃣ 总结

这套 attention 静态并行方案可以概括为：

* attention 路径具备 GQA 的张量组织方式。
* 外层任务由 `SequenceSlice` 提供 slice 级输入，而不是直接按 batch 均分。
* slice 足够长时，内部线程划分按三角工作量静态分段，而不是按行数平均分配。
* slice 较短、无法用行块覆盖所有线程时，内部改为按 `kv_head` 波次静态推进：每一波只激活尽量少的连续 `kv_head`，再把当前波次展平后的 attention-head 槽位连续分给线程。
* 内部遍历采用二维 block 结构，当前调用参数为 `row_step=1`、`col_step=8`。
* 当前数值路径已经通过 scalar `block_flash_attention` 实现逐行 causal 截断；不过 `RowVisitPlan.tail` 目前尚未在访问路径中执行。

---

# 7️⃣ 核心思想一句话

> 这套方案的核心，是用 `SequenceSlice` 提供外层任务；长 slice 用三角工作量划分线程行区间；短 slice 则按 `kv_head` 波次推进，并把当前波次展平后的 `(kv_head, local_head)` 槽位连续静态分给线程；每个槽位内部继续执行同一套 `row_step × col_step` block causal attention 计算。
