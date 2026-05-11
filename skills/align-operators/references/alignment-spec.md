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

Use simple input-to-output matching.

## Recommended Configuration

- `hidden_size = 512`
- `intermediate_size = 1360`
- `num_attention_heads = 8`
- `head_dim = 64`

The canonical token/hidden shape is lightweight:
- hidden state: `[4, 512]`
- attention Q/K/V: `[4, 8, 64]`
- MLP input: `[4, 512]`
- MLP weight: `[1360, 512]`

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
x = torch.arange(4 * 512, dtype=torch.float32).reshape(4, 512)
x = torch.zeros((4, 512))
x = torch.ones((4, 512))
x = torch.randn((4, 512)) * 0.01
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
Rust runs operator
        ↓
save output.npy
        ↓
Python runs HF operator
        ↓
compare results
```

## Operator Usage

Directly call the operator, not the full model.

Examples:
```python
y = rmsnorm(x)
y = x @ w.T
y = torch.nn.functional.silu(x)
y = torch.softmax(x, dim=-1)
```

## Attention Testing

Attention shape:
- `Q, K, V: [4, 8, 64]`

Attention scores:
- `[8, 4, 4]`

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

## Logging Format

```text
===== RMSNorm =====

shape = (4, 512)

max_abs = 2.1e-7
mean_abs = 8.4e-9
cosine = 0.99999999

PASS
```

## Project Structure

```text
alignment/
├── generate_input.py
├── compare.py
├── dump/
│   ├── input.npy
│   ├── hf_output.npy
│   └── rust_output.npy
└── rust/
    ├── linear.rs
    ├── rmsnorm.rs
    ├── silu.rs
    ├── softmax.rs
    └── attention.rs
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
