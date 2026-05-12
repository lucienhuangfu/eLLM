---
name: align-operators
description: Align Rust eLLM operators with HuggingFace FP32 operator outputs at the operator/unit-test level. Use when comparing mathematical correctness for operators such as Linear, RMSNorm, SiLU, SwiGLU, Softmax, or Attention; generating synthetic inputs; checking FP32 determinism; or debugging shape, layout, transpose, reshape, or epsilon mismatches between Rust and Python implementations.
---

# Align Operators

Use this skill for operator-level precision alignment only.

## Core Rule

Treat each operator as a mathematical function and compare outputs for the same synthetic input.

Do not use this skill for full transformer runtime, batch semantics, sequence semantics, KV cache behavior, or distributed execution.

## Fused Operators Note

Rust may implement fused operators (e.g., fused SiLU + multiply, fused add + RMS norm) for performance optimization, while HuggingFace/Python will implement them as separate operators. When aligning:
- Do not worry about implementation details (fused vs non-fused)
- Only focus on input/output mathematical equivalence
- For a fused Rust operator, generate the expected output by running the individual HF operators sequentially on the same input

## Standard Setup

- Use a realistic model configuration.
- Prefer `hidden_size=2048`, `intermediate_size=5472`, `moe_intermediate_size=768`, `num_attention_heads=32`, `num_key_value_heads=4`, `head_dim=128`, `num_experts=128`, `num_experts_per_tok=8`.
- Keep tensors lightweight but representative, such as `[4, 2048]`, `[4, 32, 128]`, and `[4, 4, 128]`.
- Use strict FP32 in both Rust and Python.
- Keep execution deterministic.

## Recommended Workflow

1. Generate synthetic inputs in Python and save them as `input.npy`.
2. Run the Rust operator (fused or not) on the same input and save `output.npy`.
3. For a fused Rust operator, run the equivalent sequence of individual HuggingFace/PyTorch operators in FP32 to generate the reference output.
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
- Fused combinations (e.g., SiLU+Multiply, Add+RMSNorm, MatMul+Sigmoid)

## Comparison Rules

- Use NumPy `.npy` for interchange.
- Compare the Rust output against the HuggingFace output for the same mathematical operation (even if implementation differs).
- For fused Rust operators, compare against sequential HF operators.
- Flag layout, reshape, transpose, softmax stability, and implicit `f64` issues.

## Project Structure

Each operator (including fused ones) has its own directory under `alignment/`:

```
alignment/
├── README.md
├── rope/
│   ├── dump/              # Test outputs (npy, bin files)
│   ├── generate_hf_rope.py      # Python reference
│   └── test_rope_alignment.py   # Comparison script
├── silu_mul/              # Fused SiLU + multiply example
│   ├── dump/
│   ├── generate_hf_silu_mul.py
│   └── test_silu_mul_alignment.py
├── linear/
├── rmsnorm/
└── [operator_name]/
```

To add a new operator, create a directory following the `rope/` pattern.

## Environment Tips

### Windows-Specific Issues

1. **Python Command**: On Windows, use `py` instead of `python` to run Python scripts:
   ```bash
   py alignment/[operator]/generate_hf_[operator].py
   ```
   This avoids environment/path issues that may occur with `python` command.

2. **Rust Build Cache**: If you encounter build errors like "failed to run custom build command" for dependencies like `serde` or `quote`:
   ```bash
   # Clean the target build directory
   rm -r target
   # Then rebuild
   cargo build
   ```

3. **PyTorch Installation**: Ensure PyTorch is installed in the environment used by the `py` command. Use `py -m pip` to install:
   ```bash
   py -m pip install torch numpy
   ```

## Reference Material

See [references/alignment-spec.md](references/alignment-spec.md) for the full alignment spec, tensor-shape matrix, thresholds, logging format, bug checklist, and detailed guide for adding new operators.
