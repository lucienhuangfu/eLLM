#!/usr/bin/env python3
"""Compare Rust per-layer outputs with HF reference all 28 layers."""
import numpy as np
import pathlib

DUMP_DIR = pathlib.Path("alignment/tokenizer/dump")

token_count = 15
hidden_size = 1024


def read_f16_bin(path, shape):
    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if len(shape) > 0:
        arr = arr.reshape(shape)
    return arr


def load_npy(path, shape):
    arr = np.load(str(path))
    return arr.reshape(shape)


def compare(name, rust_data, ref_data, tol=0.05):
    diff = np.abs(rust_data - ref_data)
    max_err = diff.max()
    mean_err = diff.mean()
    cos = np.dot(rust_data.ravel(), ref_data.ravel()) / (
        np.linalg.norm(rust_data.ravel()) * np.linalg.norm(ref_data.ravel())
    )
    status = "PASS" if cos > 0.9999 else "FAIL"
    marker = "  *** FAIL ***" if status == "FAIL" else ""
    print(f"  {name:40s} max_err={max_err:.4e} mean_err={mean_err:.4e} cos={cos:.10f} {status}{marker}")
    return max_err, cos, status


def main():
    # Compare layer outputs
    print("=" * 80)
    print("Layer outputs (full transformer layer = attn+residual+norm+mlp+residual)")
    print("=" * 80)

    diverged = None
    for layer_idx in range(28):
        rust_path = DUMP_DIR / f"rust_layer{layer_idx:02d}_output.bin"
        hf_path = DUMP_DIR / f"hf_layer{layer_idx:02d}_output.npy"

        if not rust_path.exists():
            print(f"  layer {layer_idx:2d}: missing rust dump")
            continue
        if not hf_path.exists():
            print(f"  layer {layer_idx:2d}: missing hf reference")
            continue

        rust_data = read_f16_bin(rust_path, [token_count, hidden_size])
        ref_data = load_npy(hf_path, [token_count, hidden_size])

        _, cos, status = compare(f"layer_{layer_idx:02d}", rust_data, ref_data)
        if status == "FAIL" and diverged is None:
            diverged = layer_idx

    # Compare final norm
    print()
    print("=" * 80)
    print("Final norm")
    print("=" * 80)
    rust_final = read_f16_bin(DUMP_DIR / "rust_final_norm.bin", [token_count, hidden_size])
    ref_final = load_npy(DUMP_DIR / "hf_final_norm.npy", [token_count, hidden_size])
    compare("final_norm", rust_final, ref_final)

    if diverged is not None:
        print(f"\nFirst divergence at layer {diverged}")
    else:
        print("\nAll layers match (or missing data)")


if __name__ == "__main__":
    main()
