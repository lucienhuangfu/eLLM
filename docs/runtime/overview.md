# Runtime 模块说明

---

`src/runtime` 是推理执行层的核心运行时模块，负责把「请求输入」整理成「可执行的 token 切片」，再把这些切片交给算子队列并行执行。

它主要包含三层职责：

* 输入准备：把聊天消息渲染成 prompt，再编码成 token
* 批次调度：按 `Decode` 优先规则生成本轮切片
* 线程执行：由线程池消费切片并顺序执行 operator queue

---

## 1. 模块总览

`src/runtime/mod.rs` 暴露了 runtime 侧最常用的入口：

* `BatchScheduler`：每轮生成 `prefill_list` 和 `decode_list`
* `ServingRunner`：线程池执行器，负责调度和算子执行
* `Phase`、`SequenceState`：batch 槽位状态

另外还包含以下子模块：

* `batch_sequence`：prompt 写入和生成文本解码
* `chat_template`：聊天模板渲染
* `tokenizer_loader`：加载 tiktoken tokenizer
* `slice_scheduler`：prefill 阶段的静态切片分配
* `operator`：把 runtime 切片传给具体算子
* `tensor`：运行时张量与缓存相关实现

---

## 2. 核心状态

### `SequenceState`

每个 batch 槽位对应一个 `SequenceState`，字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `phase` | 当前阶段，通常在 `Start / Prefill / Decode / Timeout / Eos` 之间变化 |
| `sequence_index` | 当前序列游标，表示下一段 token 的起点 |
| `kv_index` | KV 或已生成 token 的尾部位置 |
| `filling_length` | 还剩多少 prefill token 需要处理 |
| `notify` | 该槽位完成后通知上层的同步原语 |

### `Phase`

`Phase` 是状态机枚举，当前代码里最常见的流转是：

* `Start -> Prefill`
* `Prefill -> Decode`
* `Decode -> Eos`

---

## 3. 切片结构

### `SequenceSlice`

调度器不是直接按 batch 槽位执行，而是按 `SequenceSlice` 执行。

一个 slice 描述的是某个 batch 槽位上连续的一段 token：

| 字段 | 含义 |
| --- | --- |
| `batch_index` | 属于哪个 batch 槽位 |
| `sequence_index` | 这段 slice 在序列中的起点 |
| `token_start_index` | 在本轮扁平 token 视图中的起点 |
| `length` | 连续 token 长度 |
| `last_token_flag` | 这段 slice 的末 token 是否需要被当作结果 token 处理 |

### `DecodeList`

`DecodeList` 本质上是 `Vec<SequenceSlice>` 的包装，提供了：

* `push` / `clear`
* `total_token_count`
* `lookup_global_index`
* `walk_global_range`

它既承载 decode 轮的单 token 切片，也承载 prefill 轮的扁平 attention 切片。

---

## 4. 输入准备

### `ChatTemplate`

`ChatTemplate` 会加载 `chat_template.jinja`，然后把消息对 `[("role", "content")]` 渲染成最终 prompt。

### `load_tiktoken`

`tokenizer_loader` 会从：

* `tokenizer.json`
* `tokenizer_config.json`

构建 `tiktoken_rs::CoreBPE`，用于：

* prompt 编码
* 生成文本解码

### `BatchSequence`

`BatchSequence` 负责把 prompt token 写入底层 token buffer，并在推理结束后把 token 再解码回字符串。

它做的事情很直接：

* `write_prompts()`：把渲染后的 prompt 编码后写入指定 slot
* `decode_generated_text()`：根据 `sequence_index` 和 `kv_index` 读取生成结果并解码

---

## 5. 调度链路

### 调度入口

`ServingRunner::start()` 会创建线程池，并让线程 0 负责每轮调度：

1. 线程 0 调用 `BatchScheduler::schedule_batch()`
2. 所有线程通过 `Barrier` 同步
3. 每个线程依次执行 operator queue
4. 每个 operator 用同一轮的 `prefill_list` 和 `decode_list`

### `BatchScheduler`

`BatchScheduler` 每轮只做一件事：决定本轮是 `Decode`、`Prefill` 还是 `Idle`。

调度策略非常明确：

* 只要 batch 中存在 `Phase::Decode`，本轮就进入 decode 轮
* 否则如果存在 `Phase::Prefill`，本轮就进入 prefill 轮
* 否则进入 idle，并短暂 sleep 后重试

这里采用的是严格的单轮单模式，不会把 decode 和 prefill 混在同一轮里执行。

---

## 6. Prefill 切分

prefill 轮由 `FairTaskAllocator` 和 `SliceScheduler` 共同完成。

### `FairTaskAllocator`

它负责把 `total_tokens` 尽量平均拆给 `task_count` 个任务槽位：

* 前面的任务拿到 `base_quota + 1`
* 后面的任务拿到 `base_quota`

如果 `total_tokens < task_count`，只有前面少数线程会拿到任务。

### `SliceScheduler`

`SliceScheduler` 再把单条序列切成多个 `SequenceSlice`，按静态 token 配额分发到各线程。

因此：

* 长序列可能跨多个线程
* token 分配是静态的
* 不做运行时 stealing

---

## 7. 状态更新边界

这是 runtime 里最容易混淆的一点。

`BatchScheduler` 只负责规划切片，不直接推进 `SequenceState`。

实际的状态更新发生在别处：

* `serving/handlers.rs` 里，写入 prompt 时会把槽位切到 `Phase::Prefill`
* `operators/softmax/topk_softmax.rs` 里，prefill slice 消费和最终 token 输出会推进 `sequence_index`、`kv_index`、`filling_length`，并在需要时把状态切到 `Decode` 或 `Eos`

也就是说，runtime 的职责是“生成本轮工作单元”，而不是单独维护完整状态机。

---

## 8. 一轮执行概览

可以把一轮 runtime 执行理解为：

```text
thread 0:
    调度 batch，得到 prefill_size / decode_size / prefill_list / decode_list

所有线程:
    barrier 同步
    按 operator queue 顺序执行
    使用同一轮切片完成计算

算子阶段:
    依据 slice 和 Phase 更新 SequenceState
    必要时写回输出 token
    最终通过 notify 唤醒上层
```

---

## 9. 文件索引

* `src/runtime/mod.rs`
* `src/runtime/batch_sequence.rs`
* `src/runtime/chat_template.rs`
* `src/runtime/operator.rs`
* `src/runtime/runner.rs`
* `src/runtime/scheduler.rs`
* `src/runtime/slice_scheduler.rs`
* `src/runtime/tokenizer_loader.rs`
