# 06 · 流式实现与 vLLM 的对比

> 对应原 `serving.md` 章节：§10 流式实现与 vLLM 的对比。

## 1. vLLM 的流式粒度

vLLM 的流式是**每生成 1 个 token 就发送一次 SSE chunk**，即 1 token/chunk。推理循环每 decode 出一个 token 就立刻推送给客户端，首 token 延迟（TTFT）极低，是真正的增量流式。

## 2. 当前 eLLM 的流式实现

eLLM 复用 `SequenceState` 中已有的 `notify: Arc<Notify>`，不引入任何新字段，实现真增量流式：

### 2.1 推理侧（`TopKSoftmax::run`）

每次成功写入一个 decode token 后（包括 EOS token），立即调用 `record.notify.notify_one()`。EOS 时先将 `phase` 设为 `Phase::Eos`，再 notify。

### 2.2 服务侧（`chat_handlers::build_stream_response`）

流式路径不再等待一次性完成通知，而是进入循环：

1. `notifier.notified().await` — 等待下一个 token
2. 读取 `record.sequence_index`（topk 写入后设为当前 token 的位置）和 `record.phase`
3. 调用 `batch_sequences.decode_single_token(slot_index, token_index)` 解码单个 token
4. 立即推送一条 SSE chunk
5. 若 `phase == Eos`，发送带 `finish_reason: "stop"` 的最后一个 chunk，退出循环
6. 循环结束后调用 `reclaim_slot()` 释放槽位

### 2.3 非流式路径不变

仍然等待 EOS 的单次 notify，然后一次性 decode 全部生成文本。

## 3. 与 vLLM 的对比

| | vLLM | eLLM（当前） |
|---|---|---|
| 流式粒度 | 1 token/chunk | 1 token/chunk |
| 新增字段 | 无 | 无（复用 `Notify`） |
| TTFT | 极低 | 极低 |
| 非流式路径 | 不变 | 不变 |
