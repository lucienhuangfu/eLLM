#!/usr/bin/env python3
"""Generate batch outputs from HuggingFace Qwen3 model for alignment comparison.

Runs N diverse prompts through HF model.generate() batched together.
Outputs per-sequence token IDs as JSON for comparison with Rust eLLM.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "third_party" / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import AutoModelForCausalLM, AutoTokenizer


# Diverse prompts that work across languages/domains
DEFAULT_PROMPTS = [
    "你好，请用一句话介绍 Rust。",
    "What is the capital of France?",
    "Write a short poem about programming.",
    "解释一下什么是机器学习。",
    "How does a CPU cache work?",
    "用 Python 写一个快速排序算法。",
    "Tell me a joke about computers.",
    "什么是深度学习中的注意力机制？",
    "Explain the difference between stack and heap.",
    "Write a haiku about debugging.",
]


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported torch dtype: {name}")


def run_hf_batch(
    model_dir: pathlib.Path,
    prompts: list[str],
    max_new_tokens: int,
    torch_dtype: torch.dtype,
) -> dict:
    """Run HF model.generate() on a batch of prompts and return results."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
    ).eval()

    # Apply chat template to each prompt
    messages_batch = [
        [{"role": "user", "content": prompt}]
        for prompt in prompts
    ]

    rendered_prompts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        for msgs in messages_batch
    ]

    # Tokenize with padding (left-pad for batch generation)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    print(f"Batch size: {len(prompts)}", file=sys.stderr)
    print(f"Input lengths: {[len(ids) for ids in inputs.input_ids]}", file=sys.stderr)
    print(f"Max input length: {inputs.input_ids.shape[1]}", file=sys.stderr)

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract per-sequence results
    results = []
    for batch_idx in range(len(prompts)):
        # Find where the actual tokens start (skip left padding)
        attn = inputs.attention_mask[batch_idx]
        prompt_len = int(attn.sum().item())

        full_ids = output[batch_idx].tolist()
        prompt_ids = full_ids[:prompt_len]
        generated_ids = full_ids[prompt_len:]

        generated_text = tokenizer.decode(generated_ids)

        results.append({
            "batch_index": batch_idx,
            "prompt": prompts[batch_idx],
            "rendered_prompt": rendered_prompts[batch_idx],
            "input_ids": prompt_ids,
            "input_length": len(prompt_ids),
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
        })

    return {
        "batch_size": len(prompts),
        "max_new_tokens": max_new_tokens,
        "torch_dtype": str(torch_dtype),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/data/models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    torch_dtype = parse_dtype(args.torch_dtype)

    # Select prompts based on batch size
    prompts = DEFAULT_PROMPTS[:args.batch_size]
    while len(prompts) < args.batch_size:
        prompts.append(f"Tell me something interesting about the number {len(prompts)}.")

    result = run_hf_batch(model_dir, prompts, args.max_new_tokens, torch_dtype)

    json_output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        pathlib.Path(args.output).write_text(json_output, encoding="utf-8")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
