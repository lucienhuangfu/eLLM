# Deployment

The current deployment is a single eLLM process serving one fixed model on one
CPU host.

## Fast path

```bash
cargo build --release --bin main
./target/release/main
```

Before starting, place the complete model snapshot at
`models/Qwen3-Coder-30B-A3B-Instruct`. See [Installation](../getting_started/installation.md)
for the download command and hardware guidance.

## Readiness check

```bash
curl --fail http://localhost:8000/status
```

Do not send traffic until this succeeds: weights and the computation graph are
initialized before the HTTP listener starts.

## Process management

For a long-running host, place the command behind the process supervisor used by
your environment and restart it on failure. Preserve stdout/stderr because the
startup timing, effective threads, affinity, and per-request token counts are
reported there. Expose port 8000 only to trusted clients or put an authenticated
reverse proxy in front of it; the built-in server does not provide TLS or
authentication.

Capacity defaults to one request slot. Load-test your chosen sequence length,
batch size, memory capacity, and CPU affinity before increasing concurrency.
