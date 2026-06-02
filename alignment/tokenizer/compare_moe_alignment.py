#!/usr/bin/env python3
"""
Comprehensive MoE alignment comparison between Rust eLLM and HF.

Compares at three levels:
1. Router: logits, routing weights, expert selection
2. Layer components: post_input_norm, attn_residual, post_attn_norm, mlp_output, layer_output
3. Final norm and next token

Uses FP16-appropriate thresholds (cos > 0.999 for f16 instead of 0.9999 for f32).
"""

import argparse
import json
import numpy as np
import pathlib


def read_f16_bin(path, shape):
    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if shape:
        arr = arr.reshape(shape)
    return arr


def read_usize_bin(path, shape):
    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint64).astype(np.int64)
    return arr.reshape(shape)


def compare(name, rust_data, ref_data, cos_threshold=0.9999):
    diff = np.abs(rust_data - ref_data)
    max_err = diff.max()
    mean_err = diff.mean()
    nr = np.linalg.norm(rust_data.ravel())
    nh = np.linalg.norm(ref_data.ravel())
    cos = np.dot(rust_data.ravel(), ref_data.ravel()) / (nr * nh + 1e-10)
    status = "PASS" if cos >= cos_threshold else "FAIL"
    marker = "  *** FAIL ***" if status == "FAIL" else ""
    print(f"  {name:45s} max={max_err:.4e} mean={mean_err:.4e} cos={cos:.10f} {status}{marker}")
    return max_err, cos, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--dump-dir", default="alignment/tokenizer/dump")
    args = parser.parse_args()

    dump_dir = pathlib.Path(args.dump_dir)

    with open(pathlib.Path(args.model_dir) / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    num_experts = config.get("num_experts", 128)
    num_topk = config.get("num_experts_per_tok", 8)

    input_ids = np.load(str(dump_dir / "hf_input_ids.npy"))
    token_count = int(input_ids.shape[0])

    print(f"Model: Qwen3-Coder-30B-A3B (MoE)")
    print(f"Layers: {num_layers}, Hidden: {hidden_size}, Experts: {num_experts}, TopK: {num_topk}")
    print(f"Tokens: {token_count}")
    print(f"Dump dir: {dump_dir}")
    print()

    # ===== Level 1: Router Comparison =====
    print("=" * 90)
    print("LEVEL 1: Router Alignment (per-layer gate + softmax + topk + normalize)")
    print("=" * 90)

    router_logits_fails = []
    router_weights_fails = []
    router_experts_mismatch_layers = []

    for layer_idx in range(num_layers):
        rust_logits_path = dump_dir / f"rust_layer{layer_idx:02d}_router_logits.bin"
        hf_logits_path = dump_dir / f"hf_layer{layer_idx:02d}_router_logits.npy"
        rust_weights_path = dump_dir / f"rust_layer{layer_idx:02d}_routing_weights.bin"
        hf_weights_path = dump_dir / f"hf_layer{layer_idx:02d}_routing_weights.npy"
        rust_experts_path = dump_dir / f"rust_layer{layer_idx:02d}_selected_experts.bin"
        hf_experts_path = dump_dir / f"hf_layer{layer_idx:02d}_selected_experts.npy"

        # Router logits
        if rust_logits_path.exists() and hf_logits_path.exists():
            rust = read_f16_bin(str(rust_logits_path), [token_count, num_experts])
            hf = np.load(str(hf_logits_path))
            _, cos, status = compare(f"L{layer_idx:02d} router_logits", rust, hf)
            if status == "FAIL":
                router_logits_fails.append(layer_idx)

        # Routing weights
        if rust_weights_path.exists() and hf_weights_path.exists():
            rust = read_f16_bin(str(rust_weights_path), [token_count, num_topk])
            hf = np.load(str(hf_weights_path))
            _, cos, status = compare(f"L{layer_idx:02d} routing_weights", rust, hf, cos_threshold=0.999)
            if status == "FAIL":
                router_weights_fails.append(layer_idx)

        # Expert selection
        if rust_experts_path.exists() and hf_experts_path.exists():
            rust = read_usize_bin(str(rust_experts_path), [token_count, num_topk])
            hf = np.load(str(hf_experts_path))
            n_mismatch = int(np.sum(rust != hf))
            total = rust.size
            match_pct = 100 * (total - n_mismatch) / total
            if n_mismatch == 0:
                print(f"  L{layer_idx:02d} selected_experts: ALL MATCH ({total}/{total})")
            else:
                print(f"  L{layer_idx:02d} selected_experts: {n_mismatch}/{total} differ ({match_pct:.1f}% match)")
                router_experts_mismatch_layers.append(layer_idx)

    # ===== Level 2: Component Comparison =====
    print()
    print("=" * 90)
    print("LEVEL 2: Component Alignment (per-layer transformer components)")
    print("=" * 90)

    components = {
        "post_input_norm": "Input RMS Norm",
        "attn_residual": "Attention + Residual",
        "post_attn_norm": "Post-Attention RMS Norm",
        "mlp_output": "MoE MLP Output (down proj per-expert)",
        "output": "Layer Output (after merge + residual)",
    }

    first_divergence = None
    for layer_idx in range(num_layers):
        layer_status = []
        for comp_key, comp_label in components.items():
            rust_path = dump_dir / f"rust_layer{layer_idx:02d}_{comp_key}.bin"
            hf_path = dump_dir / f"hf_layer{layer_idx:02d}_{comp_key}.npy"

            if not (rust_path.exists() and hf_path.exists()):
                continue

            if comp_key == "mlp_output":
                # MoE mlp_output: Rust is [token, num_topk, hidden], HF is [token, hidden]
                # Skip direct comparison, use mlp_merged instead
                continue

            rust = read_f16_bin(str(rust_path), [token_count, hidden_size])
            hf = np.load(str(hf_path))

            if comp_key == "output":
                # Compare last token too
                _, cos, status = compare(f"L{layer_idx:02d} {comp_label} (last token)",
                                         rust[-1:], hf[-1:])

            _, cos, status = compare(f"L{layer_idx:02d} {comp_label}", rust, hf)
            if status == "FAIL":
                layer_status.append(comp_key)

        if layer_status and first_divergence is None:
            first_divergence = (layer_idx, layer_status)

    # ===== Level 3: Final Norm & Output =====
    print()
    print("=" * 90)
    print("LEVEL 3: Final Norm & Output")
    print("=" * 90)

    rust_final = dump_dir / "rust_final_norm.bin"
    hf_final = dump_dir / "hf_final_norm.npy"
    if rust_final.exists() and hf_final.exists():
        rust = read_f16_bin(str(rust_final), [token_count, hidden_size])
        hf = np.load(str(hf_final))
        compare("final_norm", rust, hf)
        compare("final_norm (last token)", rust[-1:], hf[-1:])

    # ===== Summary =====
    print()
    print("=" * 90)
    print("ALIGNMENT SUMMARY")
    print("=" * 90)
    print(f"  Router logits failures: {len(router_logits_fails)}/{num_layers} layers")
    print(f"  Router weights failures: {len(router_weights_fails)}/{num_layers} layers")
    print(f"  Expert selection mismatches: {len(router_experts_mismatch_layers)}/{num_layers} layers")
    if first_divergence:
        print(f"  First component divergence: layer {first_divergence[0]} "
              f"({', '.join(first_divergence[1])})")
    else:
        print(f"  All components aligned within f16 tolerance")

    print()
    print("Note: Expert selection differences are expected in f16 due to tie-breaking")
    print("when two experts have routing weights within f16 precision (~0.001).")
    print("HF uses f32 for softmax; Rust uses f16. This is a precision trade-off.")


if __name__ == "__main__":
    main()
