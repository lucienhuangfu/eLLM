# Serve

Build and run the fixed-model HTTP service:

```bash
cargo build --release --bin main
./target/release/main
```

There is currently no `serve` subcommand and no `--model` option. The model is
loaded from `models/Qwen3-Coder-30B-A3B-Instruct`, and the listener is
`0.0.0.0:8000`.

Use [Environment Variables](../configuration/env_vars.md) to tune capacity,
threads, loading, and the attention backend.
