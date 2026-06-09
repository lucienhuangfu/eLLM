#!/usr/bin/env python3
"""
Operator-level alignment test for MoE Router (Softmax + TopK + Normalize).

This tests the mathematical equivalence of:
1. Router logits: hidden @ gate_weight.T
2. Router softmax: softmax(logits, dim=-1)
3. Top-K selection: topk(probs, k)
4. Normalization: probs /= sum(probs)

At FP32 precision for both sides. Uses synthetic inputs.
"""

import numpy as np
import torch
from pathlib import Path

DUMP_DIR = Path(__file__).parent / "dump"
DUMP_DIR.mkdir(parents=True, exist_ok=True)

# Use realistic A30B config
HIDDEN_SIZE = 2048
NUM_EXPERTS = 128
NUM_TOPK = 8
NUM_TOKENS = 4  # Test with a few tokens


def generate_test_data():
    """Generate synthetic test inputs and reference outputs."""
    torch.manual_seed(42)

    # Synthetic hidden states (post_attn_norm output)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32)

    # Synthetic gate weight [num_experts, hidden_size]
    gate_weight = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.float32) * 0.02

    # Reference computation (matching HF Qwen3MoE router)
    # 1. Linear projection
    router_logits = torch.nn.functional.linear(hidden_states, gate_weight)

    # 2. Full softmax over all experts
    router_probs = torch.nn.functional.softmax(router_logits, dim=-1)

    # 3. Top-K selection
    router_top_values, router_indices = torch.topk(router_probs, NUM_TOPK, dim=-1)

    # 4. Normalize top-k probabilities
    norm_topk_values = router_top_values / router_top_values.sum(dim=-1, keepdim=True)

    return {
        'hidden_states': hidden_states,
        'gate_weight': gate_weight,
        'router_logits': router_logits,
        'router_probs': router_probs,
        'router_top_values': router_top_values,
        'router_indices': router_indices,
        'norm_topk_values': norm_topk_values,
    }


def save_test_inputs(data):
    """Save inputs that Rust can load."""
    # Save in flat float32 binary format for Rust
    hidden = data['hidden_states'].numpy().astype(np.float32)
    gate_w = data['gate_weight'].numpy().astype(np.float32)

    with open(DUMP_DIR / 'test_hidden_states.bin', 'wb') as f:
        f.write(hidden.tobytes())
    with open(DUMP_DIR / 'test_gate_weight.bin', 'wb') as f:
        f.write(gate_w.tobytes())

    # Also save as npy for Python reference
    np.save(DUMP_DIR / 'ref_router_logits.npy', data['router_logits'].numpy())
    np.save(DUMP_DIR / 'ref_router_indices.npy', data['router_indices'].numpy())
    np.save(DUMP_DIR / 'ref_router_weights.npy', data['norm_topk_values'].numpy())

    print(f"Saved test inputs to {DUMP_DIR}/")
    print(f"  hidden_states: {hidden.shape}")
    print(f"  gate_weight: {gate_w.shape}")


def main():
    data = generate_test_data()
    save_test_inputs(data)

    # Print a sample to verify
    print(f"\nToken 0 reference:")
    print(f"  Top-k experts: {data['router_indices'][0].tolist()}")
    print(f"  Top-k weights: {data['norm_topk_values'][0].tolist()}")
    print(f"  Weight sum: {data['norm_topk_values'][0].sum().item():.6f}")

    # Verify properties
    for t in range(NUM_TOKENS):
        w = data['norm_topk_values'][t]
        assert abs(w.sum().item() - 1.0) < 1e-6, f"Token {t}: weights don't sum to 1"
        # Verify descending order
        for i in range(NUM_TOPK - 1):
            assert w[i] >= w[i+1], f"Token {t}: weights not descending"


if __name__ == "__main__":
    main()
