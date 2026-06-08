# 术语表

eLLM 文档中使用的关键术语快速参考。

---

## A

**AttentionKind**  
描述一个 decoder 层注意力类型的枚举变体：`Full`、`SlidingWindow` 或 `Linear`。

---

## B

**BatchPlan**  
一轮调度的决策：`Decode`、`Prefill` 或 `Idle`。

**BatchScheduler**  
扫描 `batch_list` 并在每轮生成 `ScheduleTask`（prefill 和 decode 切片）的组件。
位于 `src/runtime/scheduling/scheduler.rs`。

**BatchSequence**  
持有所有 batch 槽位的 tokenizer 和 token 序列缓冲区。
暴露 `write_prompts()` 和 `decode_*()` 方法。

**batch_size**  
最大并发飞行中请求数。由 `ELLM_BATCH_SIZE` 控制。

---

## C

**chunk_size**  
单次 prefill 轮次处理的最大 token 数。同时用作 `TokenCounter` 的 `token_threshold`。
由 `ELLM_CHUNK_SIZE` 控制。

---

## D

**DecodeList**  
decode 轮次的 `SequenceSlice` 列表。decode 模式下每个切片长度为 1。
支持 O(log N) 的全局 token 索引查找。

**decode 轮次**  
每个活跃 `Phase::Decode` 序列各贡献一个 token 的调度轮次。

---

## F

**FfnKind**  
描述一个 decoder 层前馈网络类型的枚举变体：`Dense { intermediate_size }` 或 `SparseMoe { … }`。

**filling_length**  
batch 槽位剩余的 prefill token 数。由 `TopKSoftmax` 在每个 prefill 步骤后递减。
降为 0 时槽位转换到 `Phase::Decode`。

---

## G

**GQA（分组查询注意力）**  
多个 Q head 共享一个 KV head 的注意力变体。
比例 `num_attention_heads / num_key_value_heads` 即 `num_key_value_groups`。

---

## K

**kv_index**  
batch 槽位在 KV cache 中的下一个写入位置。

---

## L

**LayerPlan**  
记录一个 decoder 层的 `AttentionKind` 和 `FfnKind` 的逐层数据结构。
由 family resolver 生成，存储在 `ResolvedConfig.layers` 中。

---

## M

**ModelFamily**  
识别模型族的枚举：`Llama`、`Qwen`、`Mixtral`、`MiniMax`、`MiniMaxM2`、`Unknown`。
用于分发配置解析和 tensor 名称生成。

---

## P

**Phase**  
batch 槽位的生命周期状态：`Start` → `Prefill` → `Decode` → `Eos` → `Start`。

**prefill 轮次**  
处理 prompt token 的调度轮次。如果序列长度超过 `chunk_size`，
单个序列可能跨多个 prefill 轮次。

---

## R

**ResolvedConfig**  
从 `HfConfig` 派生的稳定运行时配置。不含族特有的可选字段。
传递给所有运行时组件。

**RoPE（旋转位置嵌入）**  
通过将 Q/K 投影与复数旋转因子相乘来注入位置信息的编码方案。
MiniMax-M2.5 使用部分维度 RoPE（`rotary_dim < head_dim`）。

---

## S

**ScheduleTask**  
从 `BatchScheduler` 广播到所有 `ServingRunner` 线程的 payload。
携带 `prefill_list`、`decode_list`、大小、时间戳和任务 ID。

**sequence_index**  
batch 槽位的序列 token 缓冲区中当前的读写游标。

**SequenceSlice**  
最小计算单元：`batch_index`、`sequence_index`、`token_start_index`、`length`、`last_token_flag`。

**SequenceState**  
serving 层跟踪的每槽位状态：`phase`、`sequence_index`、`kv_index`、`filling_length`、`notify`。

**ServingRunner**  
订阅广播的线程池执行器。每个 runner 线程订阅广播频道，
收到 `ScheduleTask` 时执行算子队列。

---

## T

**TensorNames / ModelTensorNames**  
每个族的名称描述对象，将逻辑 tensor 角色（如 `q_proj`）映射到
safetensors 中的实际键名。

**TokenCounter**  
追踪自上次调度触发以来写入的总 token 数。当计数超过 `chunk_size`（阈值触发）
或超时窗口到期时触发（超时触发）。

**TopKSoftmax**  
队列中的最后一个算子。执行采样、更新 `SequenceState` 并通知 serving 层。
将槽位从 `Prefill` 转换到 `Decode`，或从 `Decode` 转换到 `Eos`。
