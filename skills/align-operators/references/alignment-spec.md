# Operator Precision Alignment

This reference captures the detailed alignment contract for Rust eLLM operators and HuggingFace operators in FP32.

## Goal

Align operator outputs so that the same input produces the same output at the operator level.

Only verify mathematical correctness of operators.

Do not verify full transformer runtime, batch semantics, sequence semantics, KV cache behavior, or distributed execution.

## Core Principle

Treat each operator as a mathematical function:
- `Y = RMSNorm(X)`
- `Y = XW`
- `Y = SiLU(X)`
- `Y = Softmax(X)`
- `Y = (SiLU(X_gate) * X_up) @ W_down` (fused example)

Use simple input-to-output matching.

## Fused Operators Note

Rust may implement fused operators for performance optimization, while HuggingFace/Python will typically implement them as separate operators. Examples of fused operators in the codebase:
- `SiluMulZipMap`: fused SiLU + multiply
- `MatMulSigmoid`: fused MatMul + Sigmoid
- `ExpertsMatMulSilu`: fused experts MatMul + SiLU
- `AddRMSZipMap`: fused add + RMS norm

When aligning fused operators:
- Do not worry about implementation details (fused vs non-fused)
- Only focus on input/output mathematical equivalence
- For a fused Rust operator, generate the expected output by running the individual HF operators sequentially on the same input

## Recommended Configuration

- `hidden_size = 2048`
- `intermediate_size = 5472`
- `moe_intermediate_size = 768`
- `num_attention_heads = 32`
- `num_key_value_heads = 4`
- `head_dim = 128`
- `num_experts = 128`
- `num_experts_per_tok = 8`

The canonical token/hidden shape is lightweight:
- hidden state: `[4, 2048]`
- attention Q: `[4, 32, 128]`
- attention K/V: `[4, 4, 128]`
- MLP input: `[4, 2048]`
- MLP weight: `[5472, 2048]`
- MoE expert weight: `[768, 2048]`

## Why These Shapes

Realistic shapes help catch:
- layout bugs
- reshape bugs
- transpose mistakes
- SIMD misalignment issues

## FP32 Requirement

Python:
```python
torch.set_default_dtype(torch.float32)
model.float()
```

Disable:
- `fp16`
- `bf16`
- `tf32`

Rust:
- use `f32` only
- avoid `f64` intermediates
- avoid mixed precision

## Determinism

Python:
```python
torch.manual_seed(0)
```

Rust:
- use a fixed seed
- use deterministic execution

## Input Strategy

Use synthetic inputs only.

Recommended test inputs:
```python
x = torch.arange(4 * 2048, dtype=torch.float32).reshape(4, 2048)
x = torch.zeros((4, 2048))
x = torch.ones((4, 2048))
x = torch.randn((4, 2048)) * 0.01
x = torch.tensor([-100, -10, -1, 0, 1, 10, 100], dtype=torch.float32)
```

Purpose matrix:
| Input Type | Purpose |
|---|---|
| zeros | numerical stability |
| ones | scaling correctness |
| arange | layout correctness |
| random | general correctness |
| extreme | stability edge cases |

## Serialization

Use NumPy `.npy` because it is simple, fast, and preserves dtype and shape.

Workflow:
```text
Python generates input
        ↓
save input.npy
        ↓
Rust loads input
        ↓
Rust runs operator (fused or not)
        ↓
save output.npy
        ↓
Python runs equivalent HF operators (sequential if fused)
        ↓
compare results
```

## Operator Usage

Directly call the operator, not the full model. For fused operators, call the individual operators sequentially.

Examples (single operators):
```python
y = rmsnorm(x)
y = x @ w.T
y = torch.nn.functional.silu(x)
y = torch.softmax(x, dim=-1)
```

Example (fused operator reference):
```python
# For fused SiLU + multiply, run sequentially in Python
silu_out = torch.nn.functional.silu(x1)
y = silu_out * x2
```

## Attention Testing

Attention shape:
- `Q: [4, 32, 128]`
- `K, V: [4, 4, 128]`

Attention scores:
- `[32, 4, 4]`

## Metrics

Max absolute error:
```python
np.max(np.abs(a - b))
```

Mean absolute error:
```python
np.mean(np.abs(a - b))
```

Cosine similarity:
```python
np.dot(a.flatten(), b.flatten()) / (
    np.linalg.norm(a.flatten()) *
    np.linalg.norm(b.flatten())
)
```

## Acceptance Criteria

| Metric | Threshold |
|---|---|
| max abs | `< 1e-5` |
| mean abs | `< 1e-6` |
| cosine similarity | `> 0.999999` |

## Compare Script Shape

```python
import numpy as np

def compare(a_path, b_path):
    a = np.load(a_path)
    b = np.load(b_path)

    assert a.shape == b.shape

    diff = np.abs(a - b)
    cos = np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) *
        np.linalg.norm(b.flatten())
    )

    print("shape:", a.shape)
    print("max_abs:", diff.max())
    print("mean_abs:", diff.mean())
    print("cosine:", cos)
    print("PASS" if (
        diff.max() < 1e-5 and
        diff.mean() < 1e-6 and
        cos > 0.999999
    ) else "FAIL")
```

## Common Bugs

- layout mismatch: `[token, hidden]` vs `[hidden, token]`
- reshape errors: `[4, 512] → [4, 8, 64]`
- RMSNorm epsilon mismatch: Rust and HF differ
- softmax numerical stability: different max-subtraction order
- implicit `f64` accumulation: unintended double-precision intermediates
- for fused operators: incorrect order of operations

## Logging Format

```text
===== RMSNorm =====

shape = (4, 2048)

max_abs = 2.1e-7
mean_abs = 8.4e-9
cosine = 0.99999999

PASS
```

## Project Structure

Each operator (including fused ones) has its own directory under `alignment/`:

```text
alignment/
├── README.md
├── rope/
│   ├── dump/
│   │   ├── python_rope_basic.npy
│   │   ├── python_rope_partial.npy
│   │   ├── python_rope_yarn.npy
│   │   ├── rust_rope_basic.npy
│   │   ├── rust_rope_partial.npy
│   │   └── rust_rope_yarn.npy
│   ├── generate_hf_rope.py
│   ├── rope_alignment.py
│   ├── rope_alignment_test.rs
│   └── test_rope_alignment.py
├── silu_mul/              # Fused SiLU + multiply
│   ├── dump/
│   ├── generate_hf_silu_mul.py
│   ├── silu_mul_alignment_test.rs
│   └── test_silu_mul_alignment.py
├── linear/
│   ├── dump/
│   ├── generate_hf_linear.py
│   └── test_linear_alignment.py
├── rmsnorm/
│   ├── dump/
│   ├── generate_hf_rmsnorm.py
│   └── test_rmsnorm_alignment.py
└── [operator_name]/
    ├── dump/
    ├── generate_hf_[operator].py
    └── test_[operator]_alignment.py
```

## Adding a New Operator

To add alignment tests for a new operator (including fused ones):

1. **Create operator directory:**
   ```bash
   mkdir -p alignment/[operator_name]/dump
   ```

2. **Create Python reference implementation:**
   - File: `alignment/[operator_name]/generate_hf_[operator].py`
   - Implement the reference logic matching HuggingFace/PyTorch behavior (sequential for fused operators)
   - Save outputs to `dump/` directory

3. **Create test script:**
   - File: `alignment/[operator_name]/test_[operator]_alignment.py`
   - Generate synthetic inputs
   - Load Rust outputs from `dump/rust_[operator]_[case].npy`
   - Compare using `max_abs`, `mean_abs`, and `cosine_similarity`
   - Print PASS/FAIL results

4. **Generate Rust outputs:**
   - Create a Rust binary to export your operator's output
   - Save as npy f32 arrays to `dump/rust_[operator]_[case].npy`

5. **Run alignment test:**
   ```bash
   python alignment/[operator_name]/test_[operator]_alignment.py
   ```

## Final Principle

Require:
- same input
- same output

Do not require:
- same implementation
- same kernel
- same fusion strategy
- same execution path
