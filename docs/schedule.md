# 推理调度器说明

---

# 1️⃣ 问题背景

本文讨论推理阶段 `BatchScheduler` 的调度方式，目标是：

* 明确区分 `Prefill` 与 `Decode` 两类工作
* 在单轮调度中给出稳定、可复现的静态切片结果
* 尽量把本轮 token 数公平分散到各线程
* 为后续整条 operator queue 提供统一的切片输入

本文聚焦的是调度器本身如何扫描 batch 状态、决定本轮工作类型、生成切片列表以及推进序列状态。

就当前实现而言：

> `BatchScheduler` 每次只产出一轮调度结果，
> 并将结果组织成 `prefill_list`、`decode_list` 和 `attention_list` 三类切片结构，供 `ServingRunner` 和后续算子执行使用。

---

# 2️⃣ 核心对象

## 2.1 状态对象 `SequenceState`

每个 batch 槽位对应一个 `SequenceState`，调度最关心的字段如下：

| 字段 | 含义 |
| --- | --- |
| `phase` | 当前所处阶段，可能为 `Start / Prefill / Decode / Timeout / Eos` |
| `sequence_index` | 当前已经推进到的序列游标，也可以理解为“下一段待处理 token 的起点” |
| `kv_index` | 当前 KV 或已写入 token 的尾部位置；decode 时会以它作为下一个 token 的序列位置 |
| `filling_length` | 当前还剩多少个 prefill token 需要切分和调度 |

对 prefill 来说，剩余待处理 token 数直接由状态对象保存：

$$
remaining = filling\_length
$$

因此：

* `sequence_index` 更像是当前游标
* `kv_index` 更像是当前序列尾部位置，decode 时会从这里继续追加 token
* `filling_length` 表示当前还未完成的 prefill token 数
* `filling_length == 0` 表示这条序列当前没有待补齐的 prefill token

## 2.2 切片对象 `SequenceSlice`

调度器对外输出的最小工作单元不是单个 batch 槽位，而是一个 `SequenceSlice`。

它包含：

| 字段 | 含义 |
| --- | --- |
| `token_start_index` | 该切片在本轮扁平 token 视图中的起始偏移 |
| `batch_index` | 它属于哪个 batch 槽位 |
| `sequence_index` | 这段切片起始的序列位置 |
| `length` | 这段连续 token 区间的长度 |
| `lift_index` | 额外的提升目标下标；当前实现里普通切片固定为 `0`，只有 prefill 轮额外写入 `decode_list` 的提升切片会把它设为目标 batch 槽位索引 |

因此最小调度单元可以理解为：

```text
一个 batch 槽位中某段连续 token 区间上的子任务
```

## 2.3 三类输出列表

一次调度完成后，会得到三类结果：

### `prefill_list`

类型是 `Vec<Vec<SequenceSlice>>`。

* 外层下标对应 `thread_id`
* 内层保存该线程本轮负责的全部 prefill 切片

### `decode_list`

类型是 `Vec<SequenceSlice>`。

* 它不是按线程拆开的，而是本轮全部 decode 或 lift slice 的扁平列表
* decode 场景下每条序列本轮只会贡献 1 个 token
* decode 场景下每个切片的 `lift_index` 固定为 `0`
* prefill 场景下，这个列表还会额外承载每条序列本轮最后一个 token 的提升切片，供 `LiftVector` 使用
* 真正的线程内分配不再由调度器预先落桶，而是在执行阶段通过 `assign` 按 slice 数公平切分

### `attention_list`

类型是 `Vec<SequenceSlice>`。

* 它不是按线程拆开的列表
* 它记录的是“本轮哪些连续 sequence 区间需要参与 attention”
* prefill 场景下一条序列通常贡献较长切片
* decode 场景下一条序列通常只贡献长度为 1 的切片

---

# 3️⃣ 调度结构

## 3.1 单轮调度入口

`schedule_batch()` 每次只负责生成一轮切片结果。整体结构可以概括为：

```text
1. 根据线程数设置本轮 prefill 调度器的 task_count
2. 扫描 batch_list，判断这一轮应该做什么工作
3. 如果存在 Decode，则直接进入 decode 调度
4. 否则如果存在 Prefill，则进入 prefill 调度
5. 如果两者都没有，则短暂 sleep 后继续轮询
```

这里最关键的是，它先决定“本轮工作类型”，再决定“如何分配 token”。

## 3.2 工作类型判定

工作类型由 `next_batch_work()` 决定。

当前实现的判定规则是：

* 只要扫描到任意 `Phase::Decode`，本轮立即返回 `BatchWork::Decode`
* 如果没有 decode，但存在至少一个 `Phase::Prefill`，则返回 `BatchWork::Prefill`
* 如果两者都没有，则返回 `BatchWork::Idle`

因此当前调度器采用的是严格的单轮单模式：

* 一轮只跑 `Decode`
* 或者一轮只跑 `Prefill`
* 不会把两类工作混在同一轮里执行

## 3.3 Decode 优先策略

这套判定规则的本质就是：

```text
Decode 优先于 Prefill
```

它带来的直接效果是：

* decode 请求拥有更高的轮次优先级
* prefill 不会和 decode 争抢当前这轮的执行资源
* 后续整条 operator queue 可以在一轮内使用统一的 `prefill_size / decode_size` 语义

这也意味着当前实现不是混合调度器，而是一个先决定模式、再产出切片的轮次式调度器。

---

# 4️⃣ Decode 调度方式

## 4.1 候选收集

decode 候选由 `collect_decode_candidates()` 收集。

它会从 `batch_list` 中筛出所有 `Phase::Decode` 的记录，但最多只取前 `max_decode_size` 个。其中：

| 参数 | 含义 |
| --- | --- |
| `max_decode_size` | 当前等于 `batch_size` |

因此当前实现下：

* 一轮 decode 最多只会调度 `batch_size` 条 decode 序列

收集出的候选形式是：

```text
(batch_index, kv_index)
```

这里把 `kv_index` 直接作为 decode 切片的 `sequence_index` 使用，也就是把 decode token 放在已有 KV 之后的位置。

## 4.2 切片生成

decode 轮里，每个候选都会被直接追加成一个长度为 1 的切片。

随后调度器会对每个候选做两件事：

* 先向 `attention_list` 写入一个长度为 1 的切片
* 再向 `decode_list` 追加一个长度为 1 的切片

这里 `schedule_for_sequence()` 写入的 decode 切片具有两个当前实现上的固定特征：

* `sequence_index = kv_index`
* `lift_index = 0`

因此 decode 路径的特征非常清晰：

* 每条序列本轮只贡献 1 个 token
* `attention_list` 中的 decode 切片长度恒为 1
* `decode_list` 只是把这些单 token 请求按出现顺序串成一段扁平列表
* decode 调度本身不会回写 `SequenceState`，也不会推进 `sequence_index / kv_index / filling_length`

## 4.3 Decode 的均衡特性

因为每个 decode 候选的工作量完全一致，decode 轮的负载均衡就退化成一个非常简单的问题：

```text
把 N 个单 token 请求尽量平均分给 `thread_num` 个线程
```

因此：

* 调度阶段只负责产出扁平切片列表
* 执行阶段再通过 `assign` 把这段列表切成各线程的连续子区间
* 如果线程数多于 decode 数，多余线程自然不会分到切片

---

# 5️⃣ Prefill 调度方式

## 5.1 候选收集

prefill 候选由 `collect_prefill_candidates()` 收集。

它会遍历所有 `Phase::Prefill` 的记录，并为每条记录生成：

| 字段 | 含义 |
| --- | --- |
| `batch_index` | 属于哪个 batch 槽位 |
| `sequence_index` | 当前从哪里继续 prefill |
| `remaining` | 当前还剩多少 token 待补齐，来源于 `SequenceState.filling_length` |

同时还会累计总剩余 token 数：

$$
prefill\_total\_tokens = \sum remaining_i
$$

## 5.2 单轮容量限制

prefill 不是把所有待处理 token 无上限地一次性塞进本轮，而是受 `max_prefill_size` 限制：

$$
max\_prefill\_size = sequence\_length \times batch\_size
$$

本轮真正允许调度的 token 总量为：

$$
total\_tokens = \min(prefill\_total\_tokens, max\_prefill\_size)
$$

因此：

* 如果待处理总量没有超过窗口，则这一轮会尽量全部调度完
* 如果超过窗口，则只会先调度前 `max_prefill_size` 个 token，其余留待后续轮次

## 5.3 切片生成

对每个 prefill 候选，调度器会同时生成两类视图：

* 面向 attention 的连续区间视图，也就是写入 `attention_list`
* 面向线程执行的静态切片视图，也就是写入 `prefill_list`
* 面向 `LiftVector` 的最后 token 提升视图，也就是写入 `decode_list`

这里有一个关键细节。对于单条序列，本轮真正能写入 `attention_list` 的长度不是固定等于 `remaining`，而是：

```text
min(candidate.remaining, prefill_scheduler.remaining_tokens())
```

因此如果一条序列本来还剩 6 个 token，但全局只剩 2 个 token 配额，那么这一轮只会：

* 在 `attention_list` 中写入一个长度为 2 的切片
* 在 `prefill_list` 中只分配这 2 个 token
* 在 `decode_list` 中额外记录这 2 个 token 里的最后一个 token，供后续提升到对应 batch 槽位
* 把剩余 4 个 token 留到后续轮次

这说明 prefill 本身就是支持截断与续跑的。

## 5.4 状态推进

prefill 调度完成某条记录后，会计算：

```text
scheduled_for_record = 本条序列本轮实际分配到的 token 数
```

然后立即回写状态：

* `record.sequence_index += scheduled_for_record`
* `record.filling_length -= scheduled_for_record`
* 如果 `record.filling_length == 0`，说明这条序列本轮已经补齐，状态切到 `Phase::Decode`
* 如果 `record.filling_length > 0`，则保持 `Phase::Prefill`

同时，调度器还会为该序列本轮实际调度到的最后一个 token 生成一个长度为 1 的提升切片：

* `token_start_index` 指向这个最后 token 在本轮扁平 token 视图中的当前位置
* `sequence_index` 指向这个最后 token 的序列位置
* `lift_index` 直接等于该记录的 `batch_index`

这个提升动作由后续 `LiftVector` 算子消费，用来把最后一个 token 的结果搬运到后续 decode 更容易访问的位置。当前实现只把这类提升任务追加到扁平 `decode_list` 中，具体由哪个线程执行同样在运行时通过 `assign` 决定。

这意味着当前实现中：

* `sequence_index` 在调度阶段就会前移
* `filling_length` 也会在调度阶段同步扣减
* 下一轮若继续 prefill，会直接从更新后的游标继续切分
* 状态推进与切片生成是同一步骤的一部分，而不是执行完成后的补充动作

---

# 6️⃣ 静态分配方式

`BatchScheduler` 自己决定的是：

* 本轮做哪类工作
* 本轮最多允许多少 token 进入执行
* 每条序列本轮能占用多少 token

真正把 token 公平拆给线程的是两层小组件：

* `FairTaskAllocator`
* `SliceScheduler`

## 6.1 `FairTaskAllocator`

`FairTaskAllocator` 的职责是把总 token 数 `total_tokens` 尽量平均地拆给 `task_count` 个任务槽位。

当前规则是：

$$
base\_quota = \lfloor total\_tokens / task\_count \rfloor
$$

$$
extra\_quota = total\_tokens \bmod task\_count
$$

因此：

* 前 `extra_quota` 个任务拿 `base_quota + 1`
* 剩余任务拿 `base_quota`

例如 `11` 个 token 分给 `3` 个任务，结果就是：

```text
[4, 4, 3]
```

这也正是测试里验证的行为。

当 `total_tokens < task_count` 时，只有前 `total_tokens` 个任务会成为活跃任务，其余线程本轮不会拿到切片。

## 6.2 `SliceScheduler`

`SliceScheduler` 建立在 `FairTaskAllocator` 之上，它负责把“某条序列上的连续 token 区间”真正切成若干个 `SequenceSlice`，并推入对应线程的切片列表。

核心逻辑可以概括为：

```text
while 当前序列还有 token 且 allocator 还有全局配额:
	找到当前 task_index
	从该 task 剩余配额中尽量多取 token
	生成一个 SequenceSlice 推入 slice_list[task_index]
	sequence_cursor 向前推进
```

因此一条较长的序列可能会被切成多个 slice，分散到不同线程。例如：

* 某条 prefill 序列当前还剩 4 个 token
* 两个线程当前配额各为 2

那么结果就可能是：

* `thread 0` 拿到前 2 个 token 对应的切片
* `thread 1` 拿到后 2 个 token 对应的切片

而 `token_start_index` 会随着全局已分配 token 数连续递增，用于后续算子在扁平 token 视图中定位数据。

## 6.3 这套分配的含义

因此，当前调度器的线程均衡本质上是：

* 按 token 数量做静态均衡
* 不是按算子真实耗时做动态均衡
* 不做运行时 stealing，也不在执行阶段重新分配

这和 attention 文档里的“静态切分”思路是一致的，只是这里切分的是推理轮次上的 token 配额，而不是 attention 内部的 head 或 row 范围。

---

# 7️⃣ 单轮执行语义

从执行语义上看，一轮调度更接近下面这种结构：

```text
扫描 batch_list
	如果存在 Decode:
		本轮全部跑 Decode
	否则如果存在 Prefill:
		本轮全部跑 Prefill
	否则:
		Idle

如果本轮是 Decode:
	每条候选拿 1 个 token
	写入 attention_list
	再顺序追加到 decode_list
	不修改 SequenceState

如果本轮是 Prefill:
	先统计全部剩余 token
	再截断到 max_prefill_size
	依次处理每条序列
		写入 attention_list 中本轮实际连续区间
		再按线程配额拆到 prefill_list
		再为最后一个已调度 token 生成一条 lift slice 到 decode_list，其中 lift_index = batch_index
		立即推进 sequence_index 并扣减 filling_length
		若 filling_length 归零则切到 Decode，否则保持 Prefill
```

这说明 `BatchScheduler` 更像一层批次编排器，而不是算子本身的一部分。

---

# 8️⃣ 典型行为

## 8.1 纯 Prefill

假设：

* 8 条序列都处于 `Prefill`
* 每条序列都从 `sequence_index = 0` 开始
* 每条 `filling_length = 6`
* 每条 `kv_index = 6`
* `sequence_length = 8`
* `batch_size = 32`
* `thread_num = 8`

则：

* 总 prefill token 数为 $8 \times 6 = 48$
* `max_prefill_size = 8 \times 32 = 256`
* 因此 48 个 token 会在一轮内全部进入调度

结果表现为：

* `prefill_count = 48`
* `decode_count = 0`
* `attention_list` 中有 8 个长度为 6 的切片
* `decode_list` 中有 8 个长度为 1 的提升切片；每条切片的 `sequence_index` 指向该序列本轮最后一个 token，`lift_index` 等于对应 `batch_index`
* 各线程拿到的 prefill 工作量是均匀的
* 所有序列在本轮后切到 `Decode`，且对应 `filling_length` 变为 0

## 8.2 Decode 抢占 Prefill 轮次

如果 batch 中同时存在 `Decode` 和 `Prefill`：

* 只要有任意 decode 记录存在
* 本轮就不会安排 prefill

这体现了当前实现的固定优先级策略，也就是 decode 延迟优先。

## 8.3 Prefill 窗口截断

假设：

* `sequence_length = 4`
* `batch_size = 2`
* 所以 `max_prefill_size = 8`
* 有两条 prefill 序列，每条 `filling_length = 6`

那么：

* 总需求是 12
* 但单轮最多只能调度 8

因此可能出现：

* 第一条序列在本轮内被完整补齐并切到 `Decode`
* 第二条序列只前进 2 个 token，`filling_length` 从 6 变成 4，仍保持 `Prefill`
* 两条序列各自本轮最后一个已调度 token 都会在 `decode_list` 中生成一条提升切片
* 下一轮再从新的 `sequence_index` 继续补齐

---

# 9️⃣ 与执行器的关系

`ServingRunner` 中由线程 0 调用 `schedule_batch()`，得到：

* `prefill_size`
* `decode_size`
* `prefill_list`
* `decode_list`
* `attention_list`

随后所有线程通过 barrier 同步，再基于同一轮切片结果去执行整条 operator queue。

因此更准确地说：

* 调度器不直接做算子计算
* 它负责决定这一轮哪些 token 会进入计算
* 它把这些 token 组织成可被各线程消费的静态切片结构

---

# 🔟 总结

这套推理调度方案可以概括为：

* 外层以 `SequenceState` 维护每条序列的阶段、游标和剩余 prefill 长度
* 单轮先用 Decode 优先规则决定本轮工作模式
* Decode 场景下每条序列只分配 1 个 token，并只产出切片不回写状态
* Prefill 场景下按 `SequenceState.filling_length` 汇总剩余 token，并结合窗口上限决定本轮可处理范围
* 线程间负载由 `FairTaskAllocator` 和 `SliceScheduler` 以静态 token 配额方式完成均分
* Prefill 场景下还会额外记录每条序列本轮最后一个 token 的提升位置到 `decode_list`，这类切片的 `lift_index` 等于 `batch_index`
* 调度阶段会直接推进 `sequence_index`、扣减 `filling_length`，并在补齐后切换 `phase`

---

# 1️⃣1️⃣ 核心思想一句话

> 这套方案的核心，是先用 Decode 优先策略决定单轮工作类型；Prefill 轮把 token 静态切成 `SequenceSlice` 分配到各线程，Decode 和 lift 任务则统一保存在扁平 `decode_list` 中并在执行阶段再按 `assign` 公平切分。与此同时，Prefill 轮会在调度阶段直接推进序列游标、扣减剩余 `filling_length` 并更新阶段状态。
