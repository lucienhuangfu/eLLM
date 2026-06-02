#!/usr/bin/env python3
"""Compare Rust per-layer outputs with HF reference dumps."""
import argparse
import json
import numpy as np
import pathlib


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


def compare(name, rust_data, ref_data):
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


def compare_last_token(name, rust_data, ref_data):
    return compare(f"{name}_last_token", rust_data[-1:], ref_data[-1:])


def compare_component(dump_dir, layer_idx, component, ref_data, token_count, hidden_size):
    rust_path = dump_dir / f"rust_layer{layer_idx:02d}_{component}.bin"
    if not rust_path.exists():
        print(f"  layer_{layer_idx:02d}_{component:24s} missing rust dump")
        return None
    rust_data = read_f16_bin(rust_path, [token_count, hidden_size])
    return compare(f"layer_{layer_idx:02d}_{component}", rust_data, ref_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default="models/Qwen3-0.6B")
    parser.add_argument("--dump-dir", default="alignment/tokenizer/dump")
    args = parser.parse_args()

    dump_dir = pathlib.Path(args.dump_dir)
    with open(pathlib.Path(args.model_dir) / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    hidden_size = int(config["hidden_size"])
    num_hidden_layers = int(config["num_hidden_layers"])
    input_ids = np.load(str(dump_dir / "hf_input_ids.npy"))
    token_count = int(input_ids.shape[0])

    # Compare layer outputs
    print("=" * 80)
    print("Layer outputs (full transformer layer = attn+residual+norm+mlp+residual)")
    print("=" * 80)
    print(f"model_dir={args.model_dir}")
    print(f"token_count={token_count} hidden_size={hidden_size} layers={num_hidden_layers}")

    diverged = None
    for layer_idx in range(num_hidden_layers):
        rust_path = dump_dir / f"rust_layer{layer_idx:02d}_output.bin"
        hf_path = dump_dir / f"hf_layer{layer_idx:02d}_output.npy"

        if not rust_path.exists():
            print(f"  layer {layer_idx:2d}: missing rust dump")
            continue
        if not hf_path.exists():
            print(f"  layer {layer_idx:2d}: missing hf reference")
            continue

        rust_data = read_f16_bin(rust_path, [token_count, hidden_size])
        ref_data = load_npy(hf_path, [token_count, hidden_size])

        _, cos, status = compare(f"layer_{layer_idx:02d}", rust_data, ref_data)
        if layer_idx == num_hidden_layers - 1:
            compare_last_token(f"layer_{layer_idx:02d}", rust_data, ref_data)
        if status == "FAIL" and diverged is None:
            diverged = layer_idx

    # Compare final norm
    print()
    print("=" * 80)
    print("Final norm")
    print("=" * 80)
    rust_final_path = dump_dir / "rust_final_norm.bin"
    hf_final_path = dump_dir / "hf_final_norm.npy"
    if rust_final_path.exists() and hf_final_path.exists():
        rust_final = read_f16_bin(rust_final_path, [token_count, hidden_size])
        ref_final = load_npy(hf_final_path, [token_count, hidden_size])
        compare("final_norm", rust_final, ref_final)
        compare_last_token("final_norm", rust_final, ref_final)
    else:
        print("missing final norm dump")

    if diverged is not None:
        print()
        print("=" * 80)
        print(f"Components around first divergence (layer {diverged})")
        print("=" * 80)
        layers_to_show = sorted(set([max(0, diverged - 1), diverged]))
        for layer_idx in layers_to_show:
            layer_input_path = dump_dir / f"hf_layer{layer_idx:02d}_input.npy"
            hf_input = load_npy(layer_input_path, [token_count, hidden_size])
            compare_component(
                dump_dir,
                layer_idx,
                "post_input_norm",
                load_npy(dump_dir / f"hf_layer{layer_idx:02d}_post_input_norm.npy", [token_count, hidden_size]),
                token_count,
                hidden_size,
            )
            hf_attn_residual = hf_input + load_npy(
                dump_dir / f"hf_layer{layer_idx:02d}_attn_output.npy",
                [token_count, hidden_size],
            )
            compare_component(
                dump_dir,
                layer_idx,
                "attn_residual",
                hf_attn_residual,
                token_count,
                hidden_size,
            )
            compare_component(
                dump_dir,
                layer_idx,
                "post_attn_norm",
                load_npy(dump_dir / f"hf_layer{layer_idx:02d}_post_attn_norm.npy", [token_count, hidden_size]),
                token_count,
                hidden_size,
            )
            compare_component(
                dump_dir,
                layer_idx,
                "mlp_output",
                load_npy(dump_dir / f"hf_layer{layer_idx:02d}_mlp_output.npy", [token_count, hidden_size]),
                token_count,
                hidden_size,
            )
            rust_layer_path = dump_dir / f"rust_layer{layer_idx:02d}_output.bin"
            rust_attn_path = dump_dir / f"rust_layer{layer_idx:02d}_attn_residual.bin"
            if rust_layer_path.exists() and rust_attn_path.exists():
                rust_layer = read_f16_bin(rust_layer_path, [token_count, hidden_size])
                rust_attn = read_f16_bin(rust_attn_path, [token_count, hidden_size])
                compare(
                    f"layer_{layer_idx:02d}_mlp_merged",
                    rust_layer - rust_attn,
                    load_npy(dump_dir / f"hf_layer{layer_idx:02d}_mlp_output.npy", [token_count, hidden_size]),
                )

    if diverged is not None:
        print(f"\nFirst divergence at layer {diverged}")
    else:
        print("\nAll layers match (or missing data)")


if __name__ == "__main__":
    main()
