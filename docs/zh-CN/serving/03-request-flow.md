# 03 · 请求处理流程

> 对应原 `serving.md` 章节：§3 请求处理流程。

## 1. 处理步骤

`chat_completions()` 的处理过程：

1. 生成请求 ID（纳秒时间戳格式：`chatcmpl-{nanos}`）
2. 读取 `stream` 参数，决定流式或非流式返回
3. 调用 `assign_slot_with_messages()` 申请空闲槽位（详见 [02-state-and-slot.md](./02-state-and-slot.md)）
4. `notifier.notified().await` 异步等待推理完成
5. 从 `batch_states` 读取生成结果，调用 `decode_generated_text()` 解码
6. 调用 `reclaim_slot()` 释放槽位
7. 返回 OpenAI 风格响应

如果写入 prompt 失败（tokenization 错误），会直接返回 `500`。

## 2. 相关文档

* 槽位分配与释放：[02-state-and-slot.md](./02-state-and-slot.md)
* 响应体构造：[04-response-format.md](./04-response-format.md)
