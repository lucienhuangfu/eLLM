# Hugging Face Alignment Reference

This page describes how the repository aligns Rust behavior with a Hugging Face
style Python oracle for transformer components.

## Goal

The reference code is not part of the runtime. It is used to produce small
golden cases so Rust and Python can be compared layer by layer.

Recommended order:

1. Describe a minimal case in `tests/reference/hf/cases/*.json`
2. Generate a golden file from the Python oracle
3. Load the same golden file from a Rust test and compare the results

## Current RoPE reference

The current Rust implementation lives in [`src/transformer/rope.rs`](../../../src/transformer/rope.rs).
The Python oracle in [`tests/reference/hf/rope.py`](../../../tests/reference/hf/rope.py) mirrors the same behavior:

- base RoPE frequency generation
- partial rotary tails that stay as `1, 0`
- YaRN scaling parsing and attention scaling

## Output format

The oracle writes a JSON object with:

- `head_dim`
- `rotary_dim`
- `max_sequence_length`
- `theta`
- `attention_scaling`
- `values`

The `values` array is flattened in the same order as the Rust implementation:
for each position, emit interleaved `cos, sin` pairs, then any identity tail
channels as `1, 0`.

## Generate a golden file

On Windows PowerShell:

```powershell
.\tests\reference\hf\run_rope_reference.ps1
```

Or directly:

```powershell
py -3 .\tests\reference\hf\rope.py --case .\tests\reference\hf\cases\rope_case_min.json --output .\tests\reference\hf\golden\rope_case_min.json
```

