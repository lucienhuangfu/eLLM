import torch
import numpy as np
import math


def inv_freqs(dim, theta):
    inv_freq = []
    for i in range(0, dim, 2):
        exponent = i / dim
        inv_freq_val = 1.0 / (theta ** exponent)
        inv_freq.append(inv_freq_val)
    return np.array(inv_freq, dtype=np.float32)


def generate_rope_reference(head_dim, rotary_dim, max_sequence_length, theta, attention_scaling=1.0):
    """
    Generate RoPE reference values matching the Rust implementation in src/transformer/rope.rs
    
    Output format: for each position, emit interleaved cos, sin pairs, then identity tail (1, 0)
    """
    inv_freq = inv_freqs(rotary_dim, theta)
    out = []
    
    for pos in range(max_sequence_length):
        t = pos
        for freq in inv_freq:
            angle = t * freq
            cos_val = math.cos(angle) * attention_scaling
            sin_val = math.sin(angle) * attention_scaling
            out.append(cos_val)
            out.append(sin_val)
        
        # Add identity tail for remaining dimensions
        rotary_pairs = rotary_dim // 2
        remaining_pairs = (head_dim // 2) - rotary_pairs
        for _ in range(remaining_pairs):
            out.append(1.0 * attention_scaling)
            out.append(0.0 * attention_scaling)
    
    return np.array(out, dtype=np.float32).reshape(max_sequence_length, head_dim)


def main():
    # Standard configuration
    head_dim = 64
    rotary_dim = 64
    max_sequence_length = 16
    theta = 10000.0
    attention_scaling = 1.0
    
    print(f"===== RoPE =====")
    print(f"Generating reference with:")
    print(f"  head_dim = {head_dim}")
    print(f"  rotary_dim = {rotary_dim}")
    print(f"  max_sequence_length = {max_sequence_length}")
    print(f"  theta = {theta}")
    print(f"  attention_scaling = {attention_scaling}")
    
    # Generate HF reference
    hf_output = generate_rope_reference(head_dim, rotary_dim, max_sequence_length, theta, attention_scaling)
    
    # Save to file
    np.save('alignment/dump/hf_rope_output.npy', hf_output)
    print(f"\nSaved to alignment/dump/hf_rope_output.npy")
    print(f"Shape: {hf_output.shape}")
    
    # Also save partial rotary case
    head_dim_partial = 8
    rotary_dim_partial = 4
    max_sequence_length_partial = 2
    
    hf_output_partial = generate_rope_reference(head_dim_partial, rotary_dim_partial, max_sequence_length_partial, theta, attention_scaling)
    np.save('alignment/dump/hf_rope_output_partial.npy', hf_output_partial)
    print(f"Saved partial case to alignment/dump/hf_rope_output_partial.npy")
    print(f"Shape: {hf_output_partial.shape}")
    
    # Also save yarn scaling case
    head_dim_yarn = 8
    rotary_dim_yarn = 8
    max_sequence_length_yarn = 16
    theta_yarn = 10000.0
    attention_scaling_yarn = 1.25
    
    # For yarn, we'll implement the scaling logic
    # This is a simplified version of yarn scaling for reference
    def yarn_scale_inv_freq(inv_freq, factor=4.0):
        return inv_freq / factor
    
    inv_freq_yarn = inv_freqs(rotary_dim_yarn, theta_yarn)
    inv_freq_yarn_scaled = yarn_scale_inv_freq(inv_freq_yarn, 4.0)
    
    out_yarn = []
    for pos in range(max_sequence_length_yarn):
        t = pos
        for freq in inv_freq_yarn_scaled:
            angle = t * freq
            cos_val = math.cos(angle) * attention_scaling_yarn
            sin_val = math.sin(angle) * attention_scaling_yarn
            out_yarn.append(cos_val)
            out_yarn.append(sin_val)
    
    hf_output_yarn = np.array(out_yarn, dtype=np.float32).reshape(max_sequence_length_yarn, head_dim_yarn)
    np.save('alignment/dump/hf_rope_output_yarn.npy', hf_output_yarn)
    print(f"Saved yarn case to alignment/dump/hf_rope_output_yarn.npy")
    print(f"Shape: {hf_output_yarn.shape}")


if __name__ == "__main__":
    main()
