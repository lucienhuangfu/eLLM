# Offline Inference

Use the model-specific binary to validate model loading and generation without
the HTTP layer:

```bash
cargo build --release --bin qwen3_coder_30b_a3b
ELLM_PROMPT="What is your name?" \
ELLM_MAX_OUTPUT_TOKENS=100 \
./target/release/qwen3_coder_30b_a3b
```

It uses the same default model directory and runtime kernels as the service. A
successful run prints initialization timing, effective threads, token count,
and the decoded response. This is the recommended first diagnostic when HTTP
serving output does not match direct inference.

See [Environment Variables](../configuration/env_vars.md) for the settings that
apply only to this binary.
