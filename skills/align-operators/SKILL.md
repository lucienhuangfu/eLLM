---
name: align-operators
description: Align Rust eLLM operators with HuggingFace FP32 operator outputs at the operator/unit-test level. Use when comparing mathematical correctness for operators such as Linear, RMSNorm, SiLU, SwiGLU, Softmax, or Attention; generating synthetic inputs; checking FP32 determinism; or debugging shape, layout, transpose, reshape, or epsilon mismatches between Rust and Python implementations.
---

# Align Operators

Use this skill for operator-level precision alignment only.

## Core Rule

Treat each operator as a mathematical function and compare outputs for the same synthetic input.

Do not use this skill for full transformer runtime, batch semantics, sequence semantics, KV cache behavior, or distributed execution.

## Standard Setup

- Use a realistic model configuration.
- Prefer `hidden_size=512`, `intermediate_size=1360`, `num_attention_heads=8`, `head_dim=64`.
- Keep tensors lightweight but representative, such as `[4, 512]` and `[4, 8, 64]`.
- Use strict FP32 in both Rust and Python.
- Keep execution deterministic.

## Recommended Workflow

1. Generate synthetic inputs in Python and save them as `input.npy`.
2. Run the Rust operator on the same input and save `output.npy`.
3. Run the HuggingFace or PyTorch operator in FP32.
4. Compare shapes, max absolute error, mean absolute error, and cosine similarity.
5. Accept only when the outputs meet the FP32 thresholds.

## Input Strategy

Use synthetic inputs only:

- sequential values for layout and reshape checks
- all zeros for stability and epsilon checks
- all ones for scaling checks
- small random values for general correctness
- extreme values for softmax, SiLU, and rsqrt edge cases

## Target Operators

Focus on:

- Linear
- RMSNorm
- SiLU
- SwiGLU
- Softmax
- Attention

## Comparison Rules

- Use NumPy `.npy` for interchange.
- Compare the Rust output against the HuggingFace output for the same operator.
- Flag layout, reshape, transpose, softmax stability, and implicit `f64` issues.

## Reference Material

See [references/alignment-spec.md](references/alignment-spec.md) for the full alignment spec, tensor-shape matrix, thresholds, logging format, and bug checklist.
