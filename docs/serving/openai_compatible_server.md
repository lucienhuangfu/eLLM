# OpenAI-Compatible Chat Server

`./target/release/main` loads
`models/Qwen3-Coder-30B-A3B-Instruct` once and listens on `0.0.0.0:8000`.

## Endpoints

- `GET /status` — readiness and loaded model
- `POST /v1/chat/completions` — one OpenAI-style chat completion

The request accepts `model`, `messages`, `stream`, `temperature`,
`max_tokens`, and `top_p`. The current release applies `temperature` and
`max_tokens`; a request-specific `top_p` is accepted but not yet applied.

## Non-streaming response

With `"stream": false`, the server waits for EOS or the output limit and returns
the full text in `choices[0].message.content`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen3-Coder-30B-A3B-Instruct",
    "messages":[{"role":"user","content":"What is your name?"}],
    "stream":false,
    "max_tokens":100
  }'
```

## Streaming response

With `"stream": true`, the response uses Server-Sent Events. eLLM emits one
generated-token delta as soon as it is available; it does not wait to assemble
the complete answer. The terminal event has an empty delta and
`finish_reason: "stop"`.

For a readable terminal experience, use:

```bash
python3 scripts/chat.py "What's your name?"
```

For protocol debugging, use `curl -N` and inspect the `data:` records directly.

## Scheduling lifecycle

Each request acquires a batch slot, writes the rendered chat prompt, and enters
prefill. The runner then performs prefill and token-by-token decode. Streaming
requests are notified on each token; non-streaming requests are notified at
completion. EOS is used internally to stop generation and is not included in
assistant content. The slot is reset and returned to the free queue afterward.

The API surface is intentionally small. It is compatible with the documented
chat-completions request shape, but it is not a complete implementation of every
OpenAI API option.

## Code locations

- `src/bin/main.rs` — fixed model and service startup
- `src/serving/mod.rs` — routes and API structures
- `src/serving/chat_handlers.rs` — requests and response streaming
- `src/serving/model_setup.rs` — model, weights, EOS, and thread initialization
- `src/serving/config.rs` — capacity and scheduling defaults
