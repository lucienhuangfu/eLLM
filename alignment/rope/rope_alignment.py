import math
import numpy as np
import json


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


class RotaryEmbedding:
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

        return np.array(out, dtype=np.float32).reshape(self.max_sequence_length, self.head_dim)


def main():
    print("===== RoPE Alignment =====")

    # Test 1: Basic RoPE
    print("\n--- Test 1: Basic RoPE ---")
    rope = RotaryEmbedding(64, 64, 16, 10000.0)
    output = rope.forward()
    print(f"Output shape: {output.shape}")
    np.save("alignment/dump/hf_rope_basic.npy", output)

    # Test 2: Partial Rotary
    print("\n--- Test 2: Partial Rotary ---")
    rope_partial = RotaryEmbedding(8, 4, 2, 10000.0)
    output_partial = rope_partial.forward()
    print(f"Output shape: {output_partial.shape}")
    np.save("alignment/dump/hf_rope_partial.npy", output_partial)

    # Test 3: Yarn Scaling
    print("\n--- Test 3: Yarn Scaling ---")
    rope_scaling = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 2,
        "attention_factor": 1.25,
    }
    rope_yarn = RotaryEmbedding(8, 8, 16, 10000.0, rope_scaling=rope_scaling)
    output_yarn = rope_yarn.forward()
    print(f"Output shape: {output_yarn.shape}")
    print(f"Attention scaling: {rope_yarn.attention_scaling}")
    np.save("alignment/dump/hf_rope_yarn.npy", output_yarn)

    # Now run Rust tests to verify our implementation matches
    print("\n--- Done generating reference outputs ---")


if __name__ == "__main__":
    main()
