# Qwen3-MoE Alignment

This directory is the model-level counterpart to the operator alignment skill.
Use the operator skill for isolated math kernels. Use this flow when checking
Qwen3-MoE component and full-transformer behavior against Hugging Face.

## Levels

1. Operator: synthetic FP32 inputs under `alignment/<operator>/`.
2. Component: HF hooks dump embeddings, layer inputs, attention outputs, MoE/MLP
   outputs, post-attention norm, and layer outputs.
3. Transformer: Rust runs the same one-token prompt and dumps every layer output
   plus final norm for comparison.

## Qwen3-Coder-30B-A3B

Default model path:

```bash
/data/models/Qwen3-Coder-30B-A3B-Instruct
```

Run the whole alignment:

```bash
python3 alignment/qwen3_moe/run_alignment.py /data/models/Qwen3-Coder-30B-A3B-Instruct
```

The runner defaults the Hugging Face side to `float16` to keep 30B memory use
reasonable, and runs the Rust alignment binary in `--release` mode. Use
`--hf-dtype float32` only for small cases or machines with enough memory. Use
`--debug-rust` only when compile/debug iteration matters more than runtime.

Run only one side when iterating:

```bash
python3 alignment/qwen3_moe/run_alignment.py /data/models/Qwen3-Coder-30B-A3B-Instruct --skip-hf
python3 alignment/qwen3_moe/run_alignment.py /data/models/Qwen3-Coder-30B-A3B-Instruct --skip-rust
```

Outputs are written to `alignment/tokenizer/dump/`.

## Loader Note

The Qwen3-Coder-30B-A3B checkpoint on this machine already stores expert
weights as split `experts.{id}.gate_proj.weight`, `up_proj.weight`, and
`down_proj.weight` tensors. The alignment path uses
`load_all_weights_f16_parallel()` just like the 30B runner, and it does not
override `ELLM_LOAD_THREADS`; normal inference loading behavior is not changed.
