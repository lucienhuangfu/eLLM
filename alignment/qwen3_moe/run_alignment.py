#!/usr/bin/env python3
"""Run Qwen3-MoE alignment from HF component dumps to Rust transformer dumps."""
import argparse
import json
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = "/data/models/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_DUMP_DIR = REPO_ROOT / "alignment" / "tokenizer" / "dump"


def run(cmd, cwd=REPO_ROOT):
    print("$", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def model_summary(model_dir: pathlib.Path) -> dict:
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return {
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_experts": config.get("num_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "norm_topk_prob": config.get("norm_topk_prob"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dump-dir", default=str(DEFAULT_DUMP_DIR))
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--hf-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--debug-rust", action="store_true")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    dump_dir = pathlib.Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    if not (model_dir / "config.json").exists():
        raise SystemExit(f"missing config.json under {model_dir}")

    print(json.dumps(model_summary(model_dir), indent=2), flush=True)

    if not args.skip_hf:
        run([
            sys.executable,
            "alignment/tokenizer/dump_hf_layer_outputs.py",
            str(model_dir),
            str(dump_dir),
            "--torch-dtype",
            args.hf_dtype,
        ])

    if not args.skip_rust:
        rust_cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "qwen3_one_token_alignment",
            "--",
            str(model_dir),
        ]
        if args.debug_rust:
            rust_cmd.remove("--release")
            rust_cmd.insert(2, "--quiet")
        run(rust_cmd)

    if not args.skip_compare:
        run([
            sys.executable,
            "alignment/tokenizer/compare_layers.py",
            str(model_dir),
            "--dump-dir",
            str(dump_dir),
        ])


if __name__ == "__main__":
    main()
