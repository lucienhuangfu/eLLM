# Quickstart

This is the shortest supported path from a model snapshot to one chat request.
Complete [Installation](installation.md) first.

## 1. Check the fixed model path

```bash
test -f models/Qwen3-Coder-30B-A3B-Instruct/config.json
test -f models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json
```

The `main` binary currently serves only
`Qwen3-Coder-30B-A3B-Instruct` from this directory. It has no `--model`
argument.

## 2. Build and start the service

```bash
cargo build --release --bin main
./target/release/main
```

Model loading and initial graph construction happen once at startup and can
take some time. Keep this terminal open. The service listens on
`0.0.0.0:8000`.

From a second terminal, check readiness:

```bash
curl http://localhost:8000/status
```

```json
{"status":"running","model":"Qwen3-Coder-30B-A3B-Instruct","mode":"single_request"}
```

## 3. Send a non-streaming request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {"role": "user", "content": "What is your name?"}
    ],
    "stream": false,
    "max_tokens": 100
  }'
```

The generated text is in `choices[0].message.content`.

## 4. Stream readable text

The included client hides the SSE JSON envelope and prints text as tokens
arrive:

```bash
python3 scripts/chat.py "Write a Rust Fibonacci function."
```

To inspect the raw SSE protocol instead:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Count to five."}],
    "stream": true,
    "max_tokens": 100
  }'
```

Each `data:` event contains one generated-token delta. The last event has an
empty delta and `finish_reason: "stop"`.

## Request defaults

| Field | Default | Notes |
|---|---:|---|
| `stream` | `false` | Set to `true` for SSE |
| `temperature` | `0.7` | Applied per request |
| `max_tokens` | `100` | Applied per request |

See [Environment Variables](../configuration/env_vars.md) for server capacity,
threading, and backend settings.
