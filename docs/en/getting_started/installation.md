# Installation

## Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Rust toolchain | 1.78+ | Install via [rustup](https://rustup.rs) |
| CPU | x86-64 with AVX-512 | AVX2 scalar fallback available |
| OS | Linux or Windows | Tested on Ubuntu 22.04 and Windows 11 |
| RAM | 8 GB+ | Depends on model size |

> **Note:** eLLM is a CPU-only inference engine. No GPU is required.

---

## Build from Source

```bash
git clone <repo-url> eLLM
cd eLLM

# Debug build (faster compile, slower inference)
cargo build

# Release build (recommended for serving)
cargo build --release
```

The release profile enables `opt-level = 3`, `lto = fat`, and `codegen-units = 1`
for maximum single-threaded throughput.

---

## Download a Model

Place a HuggingFace-compatible model directory under `models/`. The minimum
required files are:

```
models/<model-name>/
├── config.json
├── generation_config.json
└── model.safetensors
```

A tokenizer (`tokenizer.json`) is also required for the serving path.

Currently tested models:

| Model | Directory name |
|-------|---------------|
| Qwen3-0.6B | `models/Qwen3-0.6B` |
| MiniMax-M2.5 | `models/MiniMax-M2.5` |
| Llama-2-7B | `models/Llama-2-7b-hf` |
| Llama-2-70B | `models/Llama-2-70b-hf` |

---

## Environment Variables

All runtime parameters are controlled via environment variables. See
[Environment Variables](../reference/env_vars.md) for the full list.

Quick defaults:

```bash
export ELLM_BATCH_SIZE=3
export ELLM_SEQUENCE_LENGTH=128
export ELLM_CHUNK_SIZE=64
export ELLM_SCHEDULE_TIMEOUT_MS=10
```

---

## Running the Server

```bash
# Serve Qwen3-0.6B on port 8000
cargo run --release --bin main -- --model models/Qwen3-0.6B

# Or with the backend binary
cargo run --release --bin backend
```

The server binds to `0.0.0.0:8000` and exposes:

- `POST /v1/chat/completions`
- `GET /status`

See [Quickstart](./quickstart.md) for example requests.
