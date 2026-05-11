import math
import numpy as np
import os
from pathlib import Path

# Get script directory to resolve relative paths
SCRIPT_DIR = Path(__file__).parent
DUMP_DIR = SCRIPT_DIR / "dump"


def inv_freqs(dim, theta):
    inv_freq = []
    for i in range(0, dim, 2):
        exponent = i / dim
        inv_freq_val = 1.0 / (theta ** exponent)
        inv_freq.append(inv_freq_val)
    return inv_freq


def yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    log_val = math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    return (dim * log_val) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    low = max(0.0, low)
    high = min(float(dim - 1), high)
    return int(math.floor(low)), int(math.ceil(high))


def yarn_linear_ramp_mask(low, high, dim):
    if abs(low - high) < 1e-9:
        high = high + 0.001

    ramp = []
    for i in range(dim):
        val = (i - low) / (high - low)
        val = max(0.0, min(1.0, val))
        ramp.append(val)
    return ramp


def apply_yarn_scaling(inv_freq, rotary_dim, theta, factor=4.0, original_max_pos=2, beta_fast=32.0, beta_slow=1.0):
    if factor <= 1.0:
        return inv_freq

    rotary_pairs = rotary_dim // 2
    if rotary_pairs == 0:
        return inv_freq

    low_rot = max(beta_fast, 1.0)
    high_rot = max(beta_slow, 1.0)
    low, high = yarn_find_correction_range(int(low_rot), int(high_rot), rotary_dim, theta, original_max_pos)
    ramp = yarn_linear_ramp_mask(float(low), float(high), rotary_pairs)
    inv_freq_extrapolation = inv_freq.copy()
    inv_freq_interpolation = [freq / factor for freq in inv_freq]

    result = []
    for i in range(rotary_pairs):
        inv_freq_mask = 1.0 - ramp[i]
        val = inv_freq_interpolation[i] * (1.0 - inv_freq_mask) + inv_freq_extrapolation[i] * inv_freq_mask
        result.append(val)
    return result


class RotaryEmbeddingPython:
    def __init__(self, head_dim, rotary_dim, max_sequence_length, theta, rope_scaling=None):
        self.head_dim = head_dim
        self.rotary_dim = min(rotary_dim, head_dim)
        self.max_sequence_length = max_sequence_length
        self.theta = theta

        self.attention_scaling = 1.0
        self.yarn_scaling = None

        if rope_scaling:
            rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
            if rope_type and rope_type.lower() == "yarn":
                factor = rope_scaling.get("factor", 1.0)
                if isinstance(factor, str):
                    factor = float(factor)
                original_max_pos = rope_scaling.get("original_max_position_embeddings", max_sequence_length)
                if isinstance(original_max_pos, str):
                    original_max_pos = int(original_max_pos)
                beta_fast = rope_scaling.get("beta_fast", 32.0)
                if isinstance(beta_fast, str):
                    beta_fast = float(beta_fast)
                beta_slow = rope_scaling.get("beta_slow", 1.0)
                if isinstance(beta_slow, str):
                    beta_slow = float(beta_slow)
                attention_factor = rope_scaling.get("attention_factor") or rope_scaling.get("attn_factor")
                if attention_factor is None:
                    attention_factor = 0.1 * math.log(factor) + 1.0
                if isinstance(attention_factor, str):
                    attention_factor = float(attention_factor)

                self.yarn_scaling = {
                    "factor": factor,
                    "original_max_position_embeddings": original_max_pos,
                    "attention_factor": attention_factor,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                }
                self.attention_scaling = attention_factor

    def forward(self):
        rotary_pairs = self.rotary_dim // 2
        inv_freq = inv_freqs(self.rotary_dim, self.theta)

        if self.yarn_scaling:
            inv_freq = apply_yarn_scaling(
                inv_freq,
                self.rotary_dim,
                self.theta,
                factor=self.yarn_scaling["factor"],
                original_max_pos=self.yarn_scaling["original_max_position_embeddings"],
                beta_fast=self.yarn_scaling["beta_fast"],
                beta_slow=self.yarn_scaling["beta_slow"],
            )

        out = []

        for pos in range(self.max_sequence_length):
            t = float(pos)
            for freq in inv_freq:
                angle = t * freq
                cos_val = math.cos(angle) * self.attention_scaling
                sin_val = math.sin(angle) * self.attention_scaling
                out.append(cos_val)
                out.append(sin_val)

            for _ in range(rotary_pairs, (self.head_dim // 2)):
                out.append(1.0 * self.attention_scaling)
                out.append(0.0 * self.attention_scaling)

        return np.array(out, dtype=np.float32)


def read_rust_npy(path):
    return np.load(path)


def compare_arrays(a, b, name):
    print(f"\n===== {name} =====")
    print(f"a shape: {a.shape}")
    print(f"b shape: {b.shape}")

    if a.shape != b.shape:
        print(f"FAIL: shape mismatch")
        return False

    diff = np.abs(a - b)
    max_abs = diff.max()
    mean_abs = diff.mean()

    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    cosine = np.dot(a_flat, b_flat) / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    print(f"\nmax_abs: {max_abs:.2e}")
    print(f"mean_abs: {mean_abs:.2e}")
    print(f"cosine: {cosine:.8f}")

    passed = max_abs < 1e-5 and mean_abs < 1e-6 and cosine > 0.999999
    print(f"\n{'PASS' if passed else 'FAIL'}")

    if not passed and len(a) < 100:
        print("\nFirst 10 elements:")
        for i in range(min(10, len(a))):
            print(f"  [{i}] Python={a[i]:.8f}, Rust={b[i]:.8f}, diff={abs(a[i]-b[i]):.2e}")

    return passed


def main():
    print("===== RoPE Alignment Test =====")

    # Generate Python reference data
    print("\n--- Generating Python reference data ---")

    # Test 1: Basic RoPE
    python_rope = RotaryEmbeddingPython(64, 64, 16, 10000.0)
    python_output = python_rope.forward()

    # Test 2: Partial Rotary
    python_rope_partial = RotaryEmbeddingPython(8, 4, 2, 10000.0)
    python_output_partial = python_rope_partial.forward()

    # Test 3: Yarn Scaling
    rope_scaling = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 2,
        "attention_factor": 1.25,
    }
    python_rope_yarn = RotaryEmbeddingPython(8, 8, 16, 10000.0, rope_scaling=rope_scaling)
    python_output_yarn = python_rope_yarn.forward()

    # Read Rust data if available, otherwise generate
    try:
        rust_output = read_rust_npy(str(DUMP_DIR / "rust_rope_basic.npy"))
        rust_output_partial = read_rust_npy(str(DUMP_DIR / "rust_rope_partial.npy"))
        rust_output_yarn = read_rust_npy(str(DUMP_DIR / "rust_rope_yarn.npy"))
    except FileNotFoundError:
        print("\nRust output files not found. Generating from Python logic only.")
        # Let's just verify our logic is correct by checking against known values
        print("\n--- Verifying Python logic against known test cases ---")

        # Test position 0 should have cos=1, sin=0
        for i in range(0, 64, 2):
            assert abs(python_output[i] - 1.0) < 1e-6
            assert abs(python_output[i + 1] - 0.0) < 1e-6
        print("✓ Position 0 check passed")

        # Save reference files
        np.save(str(DUMP_DIR / "python_rope_basic.npy"), python_output.reshape(16, 64))
        np.save(str(DUMP_DIR / "python_rope_partial.npy"), python_output_partial.reshape(2, 8))
        np.save(str(DUMP_DIR / "python_rope_yarn.npy"), python_output_yarn.reshape(16, 8))

        print(f"\nReference files saved to {DUMP_DIR}/")
        print("\nTo run full alignment:")
        print("  1. Create a simple Rust binary to export the output")
        print("  2. Or extend this script to call Rust via FFI")
        print("\nFor now, let's verify our Python reference matches the existing test expectations...")

        # Check partial rotary case
        print("\n--- Checking Partial Rotary Case ---")
        expected = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                             math.cos(1), math.sin(1), math.cos(0.01), math.sin(0.01), 1.0, 0.0, 1.0, 0.0],
                           dtype=np.float32)
        diff = np.abs(python_output_partial - expected)
        print(f"max_abs error in partial case: {diff.max():.2e}")
        if diff.max() < 1e-5:
            print("✓ Partial rotary case matches expected values")
        else:
            print("✗ Partial rotary case does not match")

        return

    # Compare if we have both
    all_passed = True
    all_passed = compare_arrays(python_output, rust_output, "RoPE (Basic)") and all_passed
    all_passed = compare_arrays(python_output_partial, rust_output_partial, "RoPE (Partial)") and all_passed
    all_passed = compare_arrays(python_output_yarn, rust_output_yarn, "RoPE (Yarn)") and all_passed

    print(f"\n{'All tests PASSED' if all_passed else 'Some tests FAILED'}")


if __name__ == "__main__":
    main()
