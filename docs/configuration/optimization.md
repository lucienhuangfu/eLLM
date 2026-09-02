# Runtime Tuning

Start with the defaults, verify correctness, and change one setting at a time.

| Goal | Suggested change | Trade-off |
|---|---|---|
| Lower graph/KV memory | Reduce `ELLM_SEQUENCE_LENGTH` | Shorter maximum prompt plus output |
| Bound one prefill round | Reduce `ELLM_CHUNK_SIZE` | More scheduling rounds for long prompts |
| Serve concurrent requests | Increase `ELLM_BATCH` | Memory grows with slots; current default is optimized for one request |
| Limit CPU use | Set `ELLM_THREAD_NUM` | Lower throughput or higher latency |
| Compare attention kernels | Set `ELLM_ATTENTION_BACKEND=native` | Disables the default BRGEMM path |
| Tune model loading | Set `ELLM_LOAD_THREADS` | Too many loaders may contend for memory bandwidth |

For a first short-context deployment:

```bash
ELLM_SEQUENCE_LENGTH=4096 \
ELLM_CHUNK_SIZE=4096 \
./target/release/main
```

For the repository defaults, simply run `./target/release/main`: batch is 1,
sequence and chunk capacity are 50,100 tokens, load workers are 16, attention
prefers BRGEMM, and inference uses all CPUs visible to the process.

The program prints `threads:` and `runner_affinity:` during initialization.
Use those lines to confirm the effective worker count and CPU placement.

See [Environment Variables](env_vars.md) for the full list and scope.
