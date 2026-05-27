#!/usr/bin/env python3
"""Compute full Qwen3-0.6B model in f16 using HF weights, compare with f32 ref."""
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


def rms_norm_f32(x, weight, eps):
    """Qwen3RMSNorm in f32 for reference."""
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x_normed)


def rms_norm_f16(x, weight, eps):
    """Qwen3RMSNorm in f16."""
    x_f16 = x.to(torch.float16)
    variance = x_f16.pow(2).mean(-1, keepdim=True)
    x_normed = x_f16 * torch.rsqrt(variance + eps)
    return (weight.to(torch.float16) * x_normed)


def layer_f32(layer, hidden_states, position_ids, cos, sin, eps, num_heads, num_kv_heads, head_dim):
    """Compute one decoder layer in f32 (reference)."""
    residual = hidden_states

    # Input layernorm
    normed = rms_norm_f32(hidden_states, layer.input_layernorm.weight.data, eps)

    # QKV projection
    q = normed @ layer.self_attn.q_proj.weight.data.T
    k = normed @ layer.self_attn.k_proj.weight.data.T
    v = normed @ layer.self_attn.v_proj.weight.data.T

    seq_len = q.shape[0]
    q = q.view(seq_len, num_heads, head_dim)
    k = k.view(seq_len, num_kv_heads, head_dim)
    v = v.view(seq_len, num_kv_heads, head_dim)

    # QK norm
    q = layer.self_attn.q_norm(q)
    k = layer.self_attn.k_norm(k)

    # RoPE
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    q_for_rope = q.unsqueeze(0).transpose(1, 2)
    k_for_rope = k.unsqueeze(0).transpose(1, 2)
    q_rotated, k_rotated = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin)
    q = q_rotated.squeeze(0)  # [16, 15, 128]
    k = k_rotated.squeeze(0)  # [8, 15, 128]

    # Attention
    scale = 1.0 / np.sqrt(head_dim)
    n_groups = num_heads // num_kv_heads
    k_expanded = k.repeat_interleave(n_groups, dim=0)
    v_transposed = v.transpose(0, 1)
    v_expanded = v_transposed.repeat_interleave(n_groups, dim=0)

    attn_scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v_expanded)
    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, -1)

    # O projection
    o_output = attn_output @ layer.self_attn.o_proj.weight.data.T

    # Residual add
    attn_final = residual + o_output

    # Post attention layernorm
    post_normed = rms_norm_f32(attn_final, layer.post_attention_layernorm.weight.data, eps)

    # MLP
    gate = post_normed @ layer.mlp.gate_proj.weight.data.T
    up = post_normed @ layer.mlp.up_proj.weight.data.T
    intermediate = torch.nn.functional.silu(gate) * up
    down = intermediate @ layer.mlp.down_proj.weight.data.T

    # Residual add
    output = attn_final + down

    return output, {
        "post_input_norm": normed,
        "attn_output": o_output,
        "post_attn_norm": post_normed,
        "mlp_output": down,
        "output": output,
    }


def layer_f16(layer, hidden_states, position_ids, cos, sin, eps, num_heads, num_kv_heads, head_dim):
    """Compute one decoder layer in f16."""
    # Convert to f16
    hs_f16 = hidden_states.to(torch.float16)
    residual = hs_f16

    # Input layernorm
    normed = rms_norm_f16(hs_f16, layer.input_layernorm.weight.data, eps)

    # QKV projection (in f16)
    q = normed @ layer.self_attn.q_proj.weight.data.to(torch.float16).T
    k = normed @ layer.self_attn.k_proj.weight.data.to(torch.float16).T
    v = normed @ layer.self_attn.v_proj.weight.data.to(torch.float16).T

    seq_len = q.shape[0]
    q = q.view(seq_len, num_heads, head_dim)
    k = k.view(seq_len, num_kv_heads, head_dim)
    v = v.view(seq_len, num_kv_heads, head_dim)

    # QK RMSNorm - HF does this in f32 internally, but we do it in f16
    q_var = q.pow(2).mean(-1, keepdim=True)
    q = q * torch.rsqrt(q_var + eps)
    q = layer.self_attn.q_norm.weight.data.to(torch.float16) * q
    k_var = k.pow(2).mean(-1, keepdim=True)
    k = k * torch.rsqrt(k_var + eps)
    k = layer.self_attn.k_norm.weight.data.to(torch.float16) * k

    # RoPE (use f32 cos/sin for accuracy, then convert)
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    q_for_rope = q.to(torch.float32).unsqueeze(0).transpose(1, 2)
    k_for_rope = k.to(torch.float32).unsqueeze(0).transpose(1, 2)
    q_rotated, k_rotated = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin)
    q = q_rotated.squeeze(0).to(torch.float16)  # [16, 15, 128]
    k = k_rotated.squeeze(0).to(torch.float16)  # [8, 15, 128]

    # Attention (in f32 for stability, then back to f16)
    scale = 1.0 / np.sqrt(head_dim)
    n_groups = num_heads // num_kv_heads
    k_expanded = k.repeat_interleave(n_groups, dim=0)
    v_transposed = v.transpose(0, 1)
    v_expanded = v_transposed.repeat_interleave(n_groups, dim=0)

    # Do attention in f32 for numerical stability (most implementations do this)
    q_f32 = q.to(torch.float32)
    k_f32 = k_expanded.to(torch.float32)
    v_f32 = v_expanded.to(torch.float32)

    attn_scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v_f32)
    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, -1).to(torch.float16)

    # O projection (in f16)
    o_output = attn_output @ layer.self_attn.o_proj.weight.data.to(torch.float16).T

    # Residual add
    attn_final = residual + o_output

    # Post attention layernorm
    post_normed = rms_norm_f16(attn_final, layer.post_attention_layernorm.weight.data, eps)

    # MLP (in f16)
    gate = post_normed @ layer.mlp.gate_proj.weight.data.to(torch.float16).T
    up = post_normed @ layer.mlp.up_proj.weight.data.to(torch.float16).T
    intermediate = torch.nn.functional.silu(gate.to(torch.float32)).to(torch.float16) * up
    down = intermediate @ layer.mlp.down_proj.weight.data.to(torch.float16).T

    # Residual add
    output = attn_final + down

    return output


def main() -> None:
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    out_dir = pathlib.Path("alignment/tokenizer/dump")
    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32,
    ).eval()

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"][0]

    embed_weight = model.model.embed_tokens.weight.data
    eps = model.config.rms_norm_eps
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    hidden_size = 1024

    # Get embeddings
    token_embeds = embed_weight[input_ids]  # [15, 1024]
    position_ids = torch.arange(len(input_ids)).unsqueeze(0)
    cos, sin = model.model.rotary_emb(token_embeds.unsqueeze(0), position_ids)

    # Compute full model in f32 and f16
    hs_f32 = token_embeds.clone()
    hs_f16 = token_embeds.to(torch.float16)

    max_errs = []

    for layer_idx, layer in enumerate(model.model.layers):
        # F32 computation
        hs_f32_new, intermediates = layer_f32(
            layer, hs_f32, position_ids, cos, sin, eps,
            num_heads, num_kv_heads, head_dim,
        )

        # F16 computation
        hs_f16_new = layer_f16(
            layer, hs_f16, position_ids, cos, sin, eps,
            num_heads, num_kv_heads, head_dim,
        )

        # Compare with HF dumped f32 reference
        hf_dump = out_dir / f"hf_layer{layer_idx:02d}_output.npy"
        if hf_dump.exists():
            hf_data = np.load(str(hf_dump))
            # Our f32 should match HF f32 exactly
            f32_diff = np.abs(hs_f32_new.detach().numpy() - hf_data)
            # Our f16 should differ slightly from HF f32
            f16_diff = np.abs(hs_f16_new.detach().to(torch.float32).numpy() - hf_data)

            # F32-to-F32 comparison
            f32_cos = np.dot(hs_f32_new.detach().numpy().ravel(), hf_data.ravel()) / (
                np.linalg.norm(hs_f32_new.detach().numpy().ravel()) * np.linalg.norm(hf_data.ravel())
            )

            # F16-to-F32 comparison
            f16_cos = np.dot(hs_f16_new.detach().to(torch.float32).numpy().ravel(), hf_data.ravel()) / (
                np.linalg.norm(hs_f16_new.detach().to(torch.float32).numpy().ravel()) * np.linalg.norm(hf_data.ravel())
            )

            max_errs.append((layer_idx, f32_diff.max(), f16_diff.max(), f32_cos, f16_cos))
            print(f"Layer {layer_idx:02d}: f32 max_err={f32_diff.max():.2e} cos={f32_cos:.10f} | "
                  f"f16_vs_f32 max_err={f16_diff.max():.2e} cos={f16_cos:.10f}")

        hs_f32 = hs_f32_new
        hs_f16 = hs_f16_new

    # Final norm
    final_norm_f32 = rms_norm_f32(hs_f32, model.model.norm.weight.data, eps)
    final_norm_f16 = rms_norm_f16(hs_f16, model.model.norm.weight.data, eps)

    # LM head
    logits_f32 = final_norm_f32 @ model.lm_head.weight.data.T
    logits_f16 = final_norm_f16 @ model.lm_head.weight.data.to(torch.float16).T

    # Get last token logits
    last_f32 = logits_f32[-1]
    last_f16 = logits_f16[-1].to(torch.float32)

    top_token_f32 = int(torch.argmax(last_f32).item())
    top_token_f16 = int(torch.argmax(last_f16).item())

    print(f"\nFinal results:")
    print(f"  f32 top token: {top_token_f32} ({tokenizer.decode([top_token_f32])})")
    print(f"  f16 top token: {top_token_f16} ({tokenizer.decode([top_token_f16])})")
    print(f"  HF reference:  151667 (<think>)")

    # Show top-5 for f16
    top5_f16 = torch.topk(last_f16, 5)
    print(f"\n  f16 top-5:")
    for i in range(5):
        tid = int(top5_f16.indices[i].item())
        print(f"    {tid}: {tokenizer.decode([tid])!r} (logit={top5_f16.values[i].item():.4f})")

    # Show top-5 for f32
    top5_f32 = torch.topk(last_f32, 5)
    print(f"\n  f32 top-5:")
    for i in range(5):
        tid = int(top5_f32.indices[i].item())
        print(f"    {tid}: {tokenizer.decode([tid])!r} (logit={top5_f32.values[i].item():.4f})")

    # Save f16 layer outputs for comparison with Rust
    print(f"\nMax error summary:")
    for layer_idx, f32_err, f16_err, f32_cos, f16_cos in max_errs:
        flag = " *** DIVERGING ***" if f16_err > 0.1 else ""
        print(f"  Layer {layer_idx:02d}: f32_err={f32_err:.2e} f16_err={f16_err:.2e}"
              f" f32_cos={f32_cos:.8f} f16_cos={f16_cos:.8f}{flag}")


if __name__ == "__main__":
    main()
