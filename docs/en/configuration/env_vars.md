# Environment Variables

All eLLM runtime parameters are read from environment variables in
`ServingConfig::new()` (`src/serving/config.rs`). Values of `0` fall back to
the built-in defaults.

---

## Serving / Scheduling

| Variable | Default | Description |
|----------|---------|-------------|
| `ELLM_BATCH_SIZE` | `3` | Maximum number of concurrent in-flight requests (batch slots) |
| `ELLM_SEQUENCE_LENGTH` | `128` | Maximum token sequence length per slot. Memory grows linearly. |
| `ELLM_CHUNK_SIZE` | `64` | Maximum tokens processed per prefill round; doubles as `token_threshold` for scheduling. |
| `ELLM_SCHEDULE_TIMEOUT_MS` | `10` | Timeout window (ms) that triggers scheduling when `token_counter > 0` even if the threshold is not reached. |

### Tuning Table

| Goal | Adjustment |
|------|-----------|
| Lower latency | Decrease `ELLM_CHUNK_SIZE` |
| Higher throughput | Increase `ELLM_CHUNK_SIZE` and `ELLM_BATCH_SIZE` |
| Longer context | Increase `ELLM_SEQUENCE_LENGTH` |
| Bursty / low-traffic | Decrease `ELLM_SCHEDULE_TIMEOUT_MS` |

---

## Thread Count

Thread counts are computed automatically by `determine_thread_config()` in
`src/serving/model_setup.rs` and are not directly user-configurable via
environment variables.

| Thread pool | Count | Formula |
|-------------|-------|---------|
| Worker threads | `max(total_cpus - async_threads, 1)` | CPU bound |
| Async threads | `2` | Tokio async executor |

---

## Model Path

The model directory is typically passed as a CLI argument (`--model <path>`) to the
main binary. See the entry point in `src/bin/main.rs` or `src/bin/backend.rs`.
