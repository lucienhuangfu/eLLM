#!/usr/bin/env python3
"""
Generate HF reference outputs for MoE router alignment test.

This implements the exact Qwen3MoE router computation:
1. Linear projection: router_logits = hidden @ gate_weight.T
2. Full softmax over all experts
3. Top-K selection
4. Normalization: selected_probs /= sum(selected_probs)

Uses synthetic inputs at FP32 for precise comparison.
"""

import numpy as np
import torch
from pathlib import Path

DUMP_DIR = Path(__file__).parent / "dump"
DUMP_DIR.mkdir(parents=True, exist_ok=True)

# Realistic Qwen3-Coder-30B-A3B config
HIDDEN_SIZE = 2048
NUM_EXPERTS = 128
NUM_TOPK = 8
NUM_TOKENS = 4


def main():
    torch.manual_seed(42)

    # Synthetic post-attention-norm hidden states
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32)

    # Synthetic router gate weight
    gate_weight = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.float32) * 0.02

    # === Reference: HF Qwen3MoE router ===
    router_logits = torch.nn.functional.linear(hidden_states, gate_weight)
    router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
    router_top_values, router_indices = torch.topk(router_probs, NUM_TOPK, dim=-1)
    # norm_topk_prob = True
    norm_topk_values = router_top_values / router_top_values.sum(dim=-1, keepdim=True)

    # Save reference outputs as npy
    np.save(DUMP_DIR / "ref_hidden_states.npy", hidden_states.numpy())
    np.save(DUMP_DIR / "ref_gate_weight.npy", gate_weight.numpy())
    np.save(DUMP_DIR / "ref_router_logits.npy", router_logits.numpy())
    np.save(DUMP_DIR / "ref_router_probs.npy", router_probs.numpy())
    np.save(DUMP_DIR / "ref_router_indices.npy", router_indices.numpy())
    np.save(DUMP_DIR / "ref_router_weights.npy", norm_topk_values.numpy())

    # Also save as flat f32 binary for Rust
    hidden_states.numpy().astype(np.float32).tofile(DUMP_DIR / "input_hidden.bin")
    gate_weight.numpy().astype(np.float32).tofile(DUMP_DIR / "input_gate_weight.bin")

    print(f"Saved reference data to {DUMP_DIR}")
    print(f"  hidden_states shape: {hidden_states.shape}")
    print(f"  gate_weight shape: {gate_weight.shape}")
    print(f"  router_logits shape: {router_logits.shape}")
    print(f"  router_weights shape: {norm_topk_values.shape}")

    # Verify
    for t in range(NUM_TOKENS):
        w = norm_topk_values[t]
        s = w.sum().item()
        print(f"  Token {t}: experts={router_indices[t].tolist()}, "
              f"weights={[f'{x:.4f}' for x in w.tolist()]}, sum={s:.6f}")
        assert abs(s - 1.0) < 1e-6, f"Token {t}: weights don't sum to 1"


if __name__ == "__main__":
    main()
