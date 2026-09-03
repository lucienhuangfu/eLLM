# Environment Variables

The defaults are intended to make the fixed Qwen3-Coder service runnable
without an export block. Set only the values needed for an experiment.

## Server and runtime

| Variable | Default | Description |
|---|---:|---|
| `ELLM_BATCH` | `1` | Number of request slots |
| `ELLM_SEQUENCE_LENGTH` | `50100` | Token capacity per slot, including prompt and output |
| `ELLM_CHUNK_SIZE` | `50100` | Maximum prefill chunk and scheduler token threshold |
| `ELLM_SCHEDULE_TIMEOUT_MS` | `10` | Low-traffic scheduling timeout in milliseconds |
| `ELLM_THREAD_NUM` | all CPUs visible to the process | Inference worker count |
| `ELLM_LOAD_THREADS` | `16` | Parallel safetensors-loading workers |
| `ELLM_ATTENTION_BACKEND` | `brgemm` | `brgemm` or `native`; BRGEMM falls back when unavailable |
| `ELLM_LIBTORCH_CPU_PATH` | auto-detected | Exact path to `libtorch_cpu.so` |

Values parsed as positive sizes fall back to their default when missing,
invalid, or zero. A 50,100-token capacity reserves substantially more memory
than a short-context configuration.

Example: reduce memory use for short requests while retaining all CPU workers:

```bash
ELLM_SEQUENCE_LENGTH=4096 ELLM_CHUNK_SIZE=4096 ./target/release/main
```

## HTTP request fields

These are JSON fields, not environment variables:

| Field | Default | Status |
|---|---:|---|
| `stream` | `false` | Supported |
| `temperature` | `0.7` | Supported per request |
| `max_tokens` | `100` | Supported per request |
| `top_p` | model generation config | Accepted by the schema; request override is not currently applied |

`max_tokens` must fit inside `ELLM_SEQUENCE_LENGTH` after tokenization of the
chat prompt.

## Offline binary only

The `qwen3_coder_30b_a3b` binary also accepts:

| Variable | Default | Description |
|---|---:|---|
| `ELLM_MODEL_DIR` | `models/Qwen3-Coder-30B-A3B-Instruct` | Model directory |
| `ELLM_PROMPT` | built-in benchmark prompt | Literal prompt text |
| `ELLM_PROMPT_REPEAT` | unset | Repeats the benchmark prompt to create a long input |
| `ELLM_MAX_OUTPUT_TOKENS` | `100` | Output limit |
| `ELLM_ALLOW_LOGICAL_THREADS` | enabled | Allows logical CPU IDs in its affinity selection |

The server `main` intentionally keeps the model path fixed and does not use
`ELLM_MODEL_DIR`.
