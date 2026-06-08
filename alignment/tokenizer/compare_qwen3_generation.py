#!/usr/bin/env python3
"""Compare Qwen3 greedy generation between eLLM and local Hugging Face."""

import argparse
import json
import pathlib
import subprocess
import sys

import torch


def load_transformers(repo_root: pathlib.Path):
    transformers_src = repo_root / "third_party" / "transformers" / "src"
    sys.path.insert(0, str(transformers_src))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer


def run_rust(repo_root: pathlib.Path, model_dir: pathlib.Path, max_new_tokens: int):
    binary = repo_root / "target" / "release" / "multi_token_alignment"
    if not binary.exists():
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "multi_token_alignment"],
            cwd=repo_root,
            check=True,
        )

    proc = subprocess.run(
        [str(binary), str(model_dir), str(max_new_tokens)],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def run_hf(repo_root: pathlib.Path, model_dir: pathlib.Path, max_new_tokens: int):
    AutoModelForCausalLM, AutoTokenizer = load_transformers(repo_root)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return {
        "prompt": prompt,
        "input_ids": inputs.input_ids[0].tolist(),
        "generated_token_ids": output[0, inputs.input_ids.shape[1] :].tolist(),
        "generated_text": tokenizer.decode(output[0, inputs.input_ids.shape[1] :]),
    }


def first_mismatch(left, right):
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return {
                "index": index,
                "rust": left_id,
                "hf": right_id,
            }
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "rust": None if len(left) <= len(right) else left[min(len(left), len(right))],
            "hf": None if len(right) <= len(left) else right[min(len(left), len(right))],
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default="models/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    model_dir = (repo_root / args.model_dir).resolve()

    rust = run_rust(repo_root, model_dir, args.max_new_tokens)
    hf = run_hf(repo_root, model_dir, args.max_new_tokens)

    input_match = rust["input_ids"] == hf["input_ids"]
    generation_match = rust["generated_token_ids"] == hf["generated_token_ids"]
    result = {
        "status": "ok" if input_match and generation_match else "mismatch",
        "model_dir": str(model_dir),
        "max_new_tokens": args.max_new_tokens,
        "input_match": input_match,
        "generation_match": generation_match,
        "first_generation_mismatch": first_mismatch(
            rust["generated_token_ids"], hf["generated_token_ids"]
        ),
        "rust_generated_token_ids": rust["generated_token_ids"],
        "hf_generated_token_ids": hf["generated_token_ids"],
        "hf_generated_text": hf["generated_text"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
