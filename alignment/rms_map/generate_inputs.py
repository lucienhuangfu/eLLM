#!/usr/bin/env python3
"""Generate synthetic inputs and reference outputs for RMSMap alignment.

All .npy files are saved as float32 for Rust npy crate compatibility.
Python computes reference using the f16-equivalent path (cast to f16,
accumulate in f32, cast back to f16), then stores as float32.
"""
import numpy as np
from pathlib import Path

DUMP_DIR = Path(__file__).parent / "dump"
DUMP_DIR.mkdir(exist_ok=True)

HIDDEN_SIZE = 1024  # Match Qwen3-0.6B
NUM_ROWS = 15       # Match the 15-token prompt


def rms_norm_f16(input_arr: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Mirror the Rust AVX-512 rms_norm kernel exactly.

    Rust does:
      1. Accumulate sum of squares in f32
      2. Compute rrms = 1/sqrt(sum/n + eps) in f32
      3. output[i] = (input[i] * rrms * weight[i]) as f16

    Returns float32 (f16-equivalent values stored in float32).
    """
    # Simulate f16 input precision by casting to float16 first
    input_f16 = input_arr.astype(np.float16)
    weight_f16 = weight.astype(np.float16)

    out = np.zeros(input_f16.shape, dtype=np.float32)
    for row in range(input_f16.shape[0]):
        # Rust accumulates sum in f32
        row_data = input_f16[row].astype(np.float32)
        row_weight = weight_f16.astype(np.float32)
        sum_sq = float((row_data * row_data).sum())
        rrms = 1.0 / np.sqrt(sum_sq / HIDDEN_SIZE + eps)
        # Rust casts each output element to f16
        out[row] = (row_data * rrms * row_weight).astype(np.float16).astype(np.float32)
    return out


def save_f32(path, arr):
    """Save array as float32 1D (npy crate only supports 1D)."""
    np.save(str(path), arr.astype(np.float32).ravel())


def main():
    eps = 1e-6
    rng = np.random.RandomState(42)

    # --- Test 1: Sequential values (1..HIDDEN_SIZE repeated) ---
    input_seq = np.tile(np.arange(1, HIDDEN_SIZE + 1, dtype=np.float32), (NUM_ROWS, 1))
    weight_ones = np.ones(HIDDEN_SIZE, dtype=np.float32)
    save_f32(DUMP_DIR / "input_seq.npy", input_seq)
    save_f32(DUMP_DIR / "weight_ones.npy", weight_ones)
    output_seq = rms_norm_f16(input_seq, weight_ones, eps)
    save_f32(DUMP_DIR / "expected_seq.npy", output_seq)

    # --- Test 2: All zeros ---
    input_zeros = np.zeros((NUM_ROWS, HIDDEN_SIZE), dtype=np.float32)
    save_f32(DUMP_DIR / "input_zeros.npy", input_zeros)
    output_zeros = rms_norm_f16(input_zeros, weight_ones, eps)
    save_f32(DUMP_DIR / "expected_zeros.npy", output_zeros)

    # --- Test 3: All ones ---
    input_ones = np.ones((NUM_ROWS, HIDDEN_SIZE), dtype=np.float32)
    save_f32(DUMP_DIR / "input_ones.npy", input_ones)
    output_ones = rms_norm_f16(input_ones, weight_ones, eps)
    save_f32(DUMP_DIR / "expected_ones.npy", output_ones)

    # --- Test 4: Random f16-like values ---
    input_rand = rng.randn(NUM_ROWS, HIDDEN_SIZE).astype(np.float16).astype(np.float32)
    save_f32(DUMP_DIR / "input_rand.npy", input_rand)
    output_rand = rms_norm_f16(input_rand, weight_ones, eps)
    save_f32(DUMP_DIR / "expected_rand.npy", output_rand)

    # --- Test 5: Random with random weight ---
    weight_rand = rng.randn(HIDDEN_SIZE).astype(np.float16).astype(np.float32)
    save_f32(DUMP_DIR / "weight_rand.npy", weight_rand)
    output_rand_w = rms_norm_f16(input_rand, weight_rand, eps)
    save_f32(DUMP_DIR / "expected_rand_w.npy", output_rand_w)

    print("Generated RMSMap test inputs and expected outputs.")
    for f in sorted(DUMP_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
