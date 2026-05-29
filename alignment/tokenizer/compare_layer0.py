#!/usr/bin/env python3
"""Compare Rust layer 0 dumps with Python reference."""
import numpy as np
import pathlib

DUMP_DIR = pathlib.Path("alignment/tokenizer/dump")
REF_DIR = pathlib.Path("alignment/matmul3/dump")

token_count = 15
hidden_size = 1024
q_cols = 2048
kv_cols = 1024


def read_f16_bin(path, shape):
    """Read raw f16 binary file."""
    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if len(shape) > 0:
        arr = arr.reshape(shape)
    return arr


def compare(name, rust_data, ref_data, tol=0.01):
    """Compare two arrays with detailed error reporting."""
    diff = np.abs(rust_data - ref_data)
    max_err = diff.max()
    mean_err = diff.mean()
    cos = np.dot(rust_data.ravel(), ref_data.ravel()) / (
        np.linalg.norm(rust_data.ravel()) * np.linalg.norm(ref_data.ravel())
    )
    status = "PASS" if max_err < tol and cos > 0.9999 else "FAIL"
    print(f"  {name}: max_err={max_err:.4e} mean_err={mean_err:.4e} cos={cos:.10f} {status}")

    if status == "FAIL":
        # Find worst token
        for tok in range(rust_data.shape[0]):
            tok_err = diff[tok].max() if len(rust_data.shape) > 1 else diff[tok]
            if tok_err > tol:
                print(f"    token[{tok}] max_err={tok_err:.4e}")
        # Show first few values
        flat_r = rust_data.ravel()
        flat_p = ref_data.ravel()
        for i in range(min(5, len(flat_r))):
            if abs(flat_r[i] - flat_p[i]) > tol:
                print(f"    [{i}]: rust={flat_r[i]:.6f} ref={flat_p[i]:.6f}")

    return max_err, cos


def main():
    # === 1. Compare normed hidden states (LookupRMSMap output) ===
    print("=== Layer 0: LookupRMSMap output (normed hidden) ===")
    rust_normed = read_f16_bin(DUMP_DIR / "rust_layer0_normed_hidden.bin", [token_count, hidden_size])
    ref_hidden = np.load(str(REF_DIR / "hidden_states.npy")).reshape(token_count, hidden_size)
    compare("normed_hidden", rust_normed, ref_hidden)

    # === 2. Compare Q output (MatMul3 Q) ===
    print("\n=== Layer 0: MatMul3 Q output ===")
    rust_q = read_f16_bin(DUMP_DIR / "rust_layer0_q_output.bin", [token_count, q_cols])
    ref_q = np.load(str(REF_DIR / "q_final.npy")).reshape(token_count, q_cols)
    compare("Q_output", rust_q, ref_q)

    # Also compare before RoPE (after QK norm but before RoPE)
    ref_q_after_norm = np.load(str(REF_DIR / "q_after_norm.npy")).reshape(token_count, -1)
    # We don't have Rust before-RoPE state, so can't compare directly

    # === 3. Compare K output (MatMul3 K) ===
    print("\n=== Layer 0: MatMul3 K output ===")
    rust_k = read_f16_bin(DUMP_DIR / "rust_layer0_k_output.bin", [token_count, kv_cols])
    ref_k = np.load(str(REF_DIR / "k_final.npy")).reshape(token_count, kv_cols)
    compare("K_output", rust_k, ref_k)

    # === 4. Compare V output (MatMul3 V) ===
    print("\n=== Layer 0: MatMul3 V output ===")
    rust_v = read_f16_bin(DUMP_DIR / "rust_layer0_v_output.bin", [token_count, kv_cols])
    ref_v = np.load(str(REF_DIR / "v_final.npy")).reshape(token_count, kv_cols)
    compare("V_output", rust_v, ref_v)

    # === 5. Quick check: is the rust data all zeros? ===
    print("\n=== Sanity checks ===")
    for name, data in [("normed_hidden", rust_normed), ("Q", rust_q), ("K", rust_k), ("V", rust_v)]:
        print(f"  {name}: min={data.min():.6f} max={data.max():.6f} mean={data.mean():.6f} nonzero={np.count_nonzero(data)}")


if __name__ == "__main__":
    main()
