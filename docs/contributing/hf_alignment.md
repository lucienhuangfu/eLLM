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

The current Rust implementation lives in [`src/transformer/rope.rs`](https://github.com/lucienhuangfu/eLLM/blob/main/src/transformer/rope.rs).
An external Python oracle can mirror the same behavior for comparison:

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

The reference generator is not included in the current repository snapshot;
add it together with its cases and goldens when introducing a new alignment
test.
