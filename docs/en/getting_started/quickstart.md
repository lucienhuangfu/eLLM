# Quickstart

This guide assumes you have already built eLLM and downloaded a model.
See [Installation](./installation.md) if you have not done that yet.

---

## 1. Start the Server

```bash
cargo run --release --bin main -- --model models/Qwen3-0.6B
```

You should see the server bind to `0.0.0.0:8000`. Confirm it is alive:

```bash
curl http://localhost:8000/status
```

Expected response:

```json
{
  "status": "running",
  "mode": "single_threaded_background_processing",
  "info": "Inference and HTTP server run on a single OS thread using current_thread runtime"
}
```

---

## 2. Send a Chat Completion Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'
```

Example response:

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
        "content": "2 + 2 = 4."
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

## 3. Streaming

Add `"stream": true` to receive Server-Sent Events with one chunk per token:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Count to 5."}],
    "stream": true
  }'
```

Each chunk is a `chat.completion.chunk` SSE event. The final chunk carries
`"finish_reason": "stop"`.

---

## 4. Sampling Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling probability |
| `max_tokens` | int | — | Max tokens to generate |
| `stream` | bool | false | Enable SSE streaming |

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

---

## 5. Using the Fake Server (No Model Weights)

For integration testing without real weights, use `fake_server`:

```bash
cargo run --release --bin fake_server
```

It uses the `FakeEcho` operator, which completes requests immediately without
running any model computation. See [FakeEcho](../operator/fake_echo.md).

---

## Next Steps

- [Serving Overview](../serving.md) — understand the full request lifecycle
- [Runtime Overview](../runtime/overview.md) — how scheduling and execution work
- [Supported Models](../models/supported_models.md) — add more models
