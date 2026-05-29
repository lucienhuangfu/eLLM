#!/usr/bin/env python3
"""Compare Rust layer outputs with manual f16 computation and HF dumps."""
import numpy as np
import pathlib
import struct

dump_dir = pathlib.Path("alignment/tokenizer/dump")

# Read Rust dumps (f16 raw bytes)
token_count = 15
hidden_size = 1024

for layer_idx in range(4):
    rust_path = dump_dir / f"rust_layer{layer_idx:02d}_output.bin"
    if not rust_path.exists():
        print(f"Layer {layer_idx:02d}: no rust dump")
        continue

    with open(rust_path, "rb") as f:
        raw = f.read()
    rust_f16 = np.frombuffer(raw, dtype=np.float16).reshape(token_count, hidden_size).astype(np.float32)

    # Read HF dump
    hf_path = dump_dir / f"hf_layer{layer_idx:02d}_output.npy"
    if hf_path.exists():
        hf_data = np.load(str(hf_path))
        diff = np.abs(rust_f16 - hf_data)
        cos = np.dot(rust_f16.ravel(), hf_data.ravel()) / (
            np.linalg.norm(rust_f16.ravel()) * np.linalg.norm(hf_data.ravel())
        )
        print(f"Layer {layer_idx:02d} vs HF f32: max_err={diff.max():.4e} mean_err={diff.mean():.4e} cos={cos:.10f}")
        # Per-token errors
        for tok in range(token_count):
            tok_err = diff[tok].max()
            if tok_err > 1.0:
                print(f"  token[{tok}] max_err={tok_err:.4e}")

    # Also show first few values
    print(f"  rust[0,:5] = {rust_f16[0,:5]}")
    if hf_path.exists():
        print(f"  hf[0,:5]   = {hf_data[0,:5]}")
    print()
