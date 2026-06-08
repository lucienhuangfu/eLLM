# 快速入门

本指南假设你已经完成了构建并下载了模型。
如果还没有，请先查看[安装](./installation.md)。

---

## 1. 启动服务

```bash
cargo run --release --bin main -- --model models/Qwen3-0.6B
```

服务将绑定到 `0.0.0.0:8000`。验证服务是否正常运行：

```bash
curl http://localhost:8000/status
```

预期响应：

```json
{
  "status": "running",
  "mode": "single_threaded_background_processing",
  "info": "Inference and HTTP server run on a single OS thread using current_thread runtime"
}
```

---

## 2. 发送 Chat Completion 请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "2 加 2 等于多少？"}
    ]
  }'
```

示例响应：

```json
{
  "id": "chatcmpl-1749500000000000",
  "object": "chat.completion",
  "created": 1749500000,
  "model": "Qwen3-0.6B",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4。"
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

## 3. 流式输出

添加 `"stream": true` 可以通过 SSE 接收逐 token 输出：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "从 1 数到 5。"}],
    "stream": true
  }'
```

每个 chunk 是一个 `chat.completion.chunk` SSE 事件。
最后一个 chunk 携带 `"finish_reason": "stop"`。

---

## 4. 采样参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 1.0 | 采样温度 |
| `top_p` | float | 1.0 | Nucleus 采样概率 |
| `max_tokens` | int | — | 最大生成 token 数 |
| `stream` | bool | false | 启用 SSE 流式输出 |

---

## 5. 使用 Fake Server（无需模型权重）

用于集成测试，不需要真实权重：

```bash
cargo run --release --bin fake_server
```

它使用 `FakeEcho` 算子，直接完成请求而不运行任何模型计算。
详见 [FakeEcho](../operator/fake_echo.md)。

---

## 下一步

- [Serving 模块说明](../serving.md) — 了解完整的请求生命周期
- [Runtime 模块总览](../runtime/overview.md) — 调度与执行机制
- [支持的模型](../models/supported_models.md) — 添加更多模型
