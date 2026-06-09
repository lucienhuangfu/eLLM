#!/usr/bin/env python3
"""Compare batch generation between eLLM (Rust) and HuggingFace for Qwen3 models.

Runs HF batch generation and Rust multi_batch_alignment on the same prompts,
then compares per-sequence generated token IDs.
"""

import argparse
import json
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def run_rust_batch(
    repo_root: pathlib.Path,
    model_dir: pathlib.Path,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    """Run Rust multi_batch_alignment binary and return parsed JSON output."""
    binary = repo_root / "target" / "release" / "multi_batch_alignment"
    if not binary.exists():
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "multi_batch_alignment"],
            cwd=repo_root,
            check=True,
        )

    proc = subprocess.run(
        [str(binary), str(model_dir), str(max_new_tokens), str(batch_size)],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def run_hf_batch(
    repo_root: pathlib.Path,
    model_dir: pathlib.Path,
    batch_size: int,
    max_new_tokens: int,
    torch_dtype: str = "float16",
) -> dict:
    """Run HF batch generation script and return parsed JSON output."""
    script = repo_root / "alignment" / "tokenizer" / "generate_hf_batch.py"

    proc = subprocess.run(
        [
            sys.executable, str(script),
            "--model-dir", str(model_dir),
            "--batch-size", str(batch_size),
            "--max-new-tokens", str(max_new_tokens),
            "--torch-dtype", torch_dtype,
        ],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def compare_batch(rust_result: dict, hf_result: dict) -> dict:
    """Compare per-sequence generated tokens between Rust and HF."""
    rust_results = {r["batch_index"]: r for r in rust_result["results"]}
    hf_results = {r["batch_index"]: r for r in hf_result["results"]}

    per_sequence = []
    all_match = True
    total_rust_tokens = 0
    total_hf_tokens = 0
    matching_tokens = 0

    for batch_idx in sorted(rust_results.keys()):
        rust_tokens = rust_results[batch_idx]["generated_token_ids"]
        hf_tokens = hf_results[batch_idx]["generated_token_ids"]

        total_rust_tokens += len(rust_tokens)
        total_hf_tokens += len(hf_tokens)

        # Find first mismatch
        first_mismatch = None
        for i, (rt, ht) in enumerate(zip(rust_tokens, hf_tokens)):
            if rt == ht:
                matching_tokens += 1
            elif first_mismatch is None:
                first_mismatch = {
                    "token_index": i,
                    "rust_token_id": rt,
                    "hf_token_id": ht,
                }

        # Count matches among shared length
        min_len = min(len(rust_tokens), len(hf_tokens))
        seq_match = (
            rust_tokens[:min_len] == hf_tokens[:min_len]
            and len(rust_tokens) == len(hf_tokens)
        )

        rust_text = rust_results[batch_idx].get("generated_text", "")
        hf_text = hf_results[batch_idx].get("generated_text", "")

        per_sequence.append({
            "batch_index": batch_idx,
            "prompt": rust_results[batch_idx].get("prompt", ""),
            "input_match": rust_results[batch_idx]["input_ids"] == hf_results[batch_idx]["input_ids"],
            "generation_match": seq_match,
            "rust_token_count": len(rust_tokens),
            "hf_token_count": len(hf_tokens),
            "matching_tokens": sum(1 for rt, ht in zip(rust_tokens[:min_len], hf_tokens[:min_len]) if rt == ht),
            "first_mismatch": first_mismatch,
            "rust_generated_text": rust_text,
            "hf_generated_text": hf_text,
        })

        if not seq_match:
            all_match = False

    return {
        "overall_match": all_match,
        "total_rust_tokens": total_rust_tokens,
        "total_hf_tokens": total_hf_tokens,
        "total_matching_tokens": matching_tokens,
        "match_rate": matching_tokens / max(total_rust_tokens, 1),
        "per_sequence": per_sequence,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/data/models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-rust", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    model_dir = pathlib.Path(args.model_dir).resolve()

    print(f"=" * 80)
    print(f"Batch Generation Alignment Test")
    print(f"Model: {model_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Torch dtype: {args.torch_dtype}")
    print(f"=" * 80)

    hf_result = None
    rust_result = None

    if not args.skip_hf:
        print("\n>>> Running HF batch generation...")
        hf_result = run_hf_batch(
            repo_root, model_dir, args.batch_size, args.max_new_tokens, args.torch_dtype
        )
        for r in hf_result["results"]:
            print(f"  HF batch {r['batch_index']}: {len(r['generated_token_ids'])} tokens -> {r['generated_text'][:80]}...")

    if not args.skip_rust:
        print("\n>>> Running Rust batch generation...")
        rust_result = run_rust_batch(
            repo_root, model_dir, args.batch_size, args.max_new_tokens
        )
        for r in rust_result["results"]:
            print(f"  Rust batch {r['batch_index']}: {len(r['generated_token_ids'])} tokens -> {r['generated_text'][:80]}...")

    if hf_result is not None and rust_result is not None:
        print("\n>>> Comparing results...")
        comparison = compare_batch(rust_result, hf_result)

        print(f"\n{'=' * 80}")
        print(f"COMPARISON SUMMARY")
        print(f"{'=' * 80}")
        print(f"  Overall match: {comparison['overall_match']}")
        print(f"  Token match rate: {comparison['match_rate']:.2%}")
        print(f"  Total matching tokens: {comparison['total_matching_tokens']}/{comparison['total_rust_tokens']}")

        for seq in comparison["per_sequence"]:
            status = "MATCH" if seq["generation_match"] else "MISMATCH"
            marker = "  ***" if not seq["generation_match"] else ""
            print(f"  Batch {seq['batch_index']}: {status} "
                  f"({seq['matching_tokens']}/{seq['rust_token_count']} tokens match){marker}")
            if seq["first_mismatch"]:
                fm = seq["first_mismatch"]
                print(f"    First mismatch at token {fm['token_index']}: "
                      f"Rust={fm['rust_token_id']}, HF={fm['hf_token_id']}")

        # Output full comparison as JSON
        output = {
            "model_dir": str(model_dir),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "comparison": comparison,
        }
        json_output = json.dumps(output, ensure_ascii=False, indent=2)
        print(f"\n{json_output}")

        if not comparison["overall_match"]:
            raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
