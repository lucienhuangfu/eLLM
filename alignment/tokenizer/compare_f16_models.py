#!/usr/bin/env python3
"""Compare HF model in f32 vs f16 vs f16-with-f32-norms."""
import json
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "third_party" / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]

    # Test with f32 model
    print("=== f32 model ===")
    model_f32 = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        out_f32 = model_f32(**inputs)
        top_f32 = int(torch.argmax(out_f32.logits[0, -1, :]).item())
    print(f"  top token: {top_f32} ({tokenizer.decode([top_f32])!r})")

    # Test with f16 model
    print("=== f16 model ===")
    model_f16 = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16,
    ).eval()
    with torch.no_grad():
        out_f16 = model_f16(**inputs)
        top_f16 = int(torch.argmax(out_f16.logits[0, -1, :].to(torch.float32)).item())
    print(f"  top token: {top_f16} ({tokenizer.decode([top_f16])!r})")

    # Compare logits for the f32 top token
    f32_logit_for_f16_top = out_f32.logits[0, -1, top_f16].item()
    f16_logit_for_f16_top = out_f16.logits[0, -1, top_f16].item()
    print(f"  f32 logit at token {top_f16}: {f32_logit_for_f16_top:.4f}")
    print(f"  f16 logit at token {top_f16}: {f16_logit_for_f16_top:.4f}")

    # Top-5 comparison
    print("\n=== Top-5 comparison ===")
    probs_f32 = torch.softmax(out_f32.logits[0, -1, :], dim=-1)
    probs_f16 = torch.softmax(out_f16.logits[0, -1, :].to(torch.float32), dim=-1)

    top5_f32 = torch.topk(probs_f32, 5)
    top5_f16 = torch.topk(probs_f16, 5)

    print("f32 top-5:")
    for i in range(5):
        tid = int(top5_f32.indices[i].item())
        print(f"  {tid}: {tokenizer.decode([tid])!r} (prob={top5_f32.values[i].item():.6f})")

    print("f16 top-5:")
    for i in range(5):
        tid = int(top5_f16.indices[i].item())
        print(f"  {tid}: {tokenizer.decode([tid])!r} (prob={top5_f16.values[i].item():.6f})")

    # Cosine similarity between f32 and f16 logits
    logits_f32 = out_f32.logits[0, -1, :].to(torch.float32)
    logits_f16 = out_f16.logits[0, -1, :].to(torch.float32)
    cos = torch.dot(logits_f32, logits_f16) / (
        torch.norm(logits_f32) * torch.norm(logits_f16)
    )
    print(f"\nLogits cosine similarity: {cos.item():.10f}")

    # Show rank of <think> token in f16
    think_id = 151667
    f16_think_logit = out_f16.logits[0, -1, think_id].item()
    f32_think_logit = out_f32.logits[0, -1, think_id].item()
    print(f"\nToken 151667 (<think>):")
    print(f"  f32 logit: {f32_think_logit:.4f}")
    print(f"  f16 logit: {f16_think_logit:.4f}")

    # Rank of <think> in f16
    sorted_logits = torch.argsort(out_f16.logits[0, -1, :].to(torch.float32), descending=True)
    think_rank_f16 = (sorted_logits == think_id).nonzero(as_tuple=True)[0].item() + 1
    print(f"  rank in f16: {think_rank_f16}")

    # Also check Rust's output token (14582) rank in f16
    rust_token = 14582
    f16_rust_logit = out_f16.logits[0, -1, rust_token].item()
    f32_rust_logit = out_f32.logits[0, -1, rust_token].item()
    rust_rank_f16 = (sorted_logits == rust_token).nonzero(as_tuple=True)[0].item() + 1
    print(f"\nToken {rust_token} (Rust output):")
    print(f"  f32 logit: {f32_rust_logit:.4f}")
    print(f"  f16 logit: {f16_rust_logit:.4f}")
    print(f"  rank in f16: {rust_rank_f16}")

    # Also check what the Rust binary currently outputs by running it
    print("\n=== Summary ===")
    print(f"HF f32: {top_f32} ({tokenizer.decode([top_f32])!r})")
    print(f"HF f16: {top_f16} ({tokenizer.decode([top_f16])!r})")


if __name__ == "__main__":
    main()
