#!/usr/bin/env python3
"""Compare Rust and HF first-layer outputs."""
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


def rms_norm(x, weight, eps):
    """Qwen3RMSNorm in f32."""
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x_normed).detach().numpy()


def main() -> None:
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
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
    input_ids = inputs["input_ids"][0]  # [seq_len]

    # Get embeddings and first layer weights
    embed_weight = model.model.embed_tokens.weight.data  # [vocab, hidden]
    layer0 = model.model.layers[0]

    # Step 1: embedding lookup
    token_embeds = embed_weight[input_ids]  # [15, 1024]

    # Step 2: RMSNorm (using layer 0's input_layernorm)
    norm_weight = layer0.input_layernorm.weight.data
    eps = model.config.rms_norm_eps
    normed_embeds = rms_norm(token_embeds, norm_weight, eps)

    # Step 3: Check against HF layer0_post_input_norm dump
    out_dir = pathlib.Path("alignment/tokenizer/dump")
    if (out_dir / "hf_layer00_post_input_norm.npy").exists():
        hf_data = np.load(str(out_dir / "hf_layer00_post_input_norm.npy"))
        diff = np.abs(normed_embeds - hf_data)
        print(f"RMSNorm: shape={normed_embeds.shape}")
        print(f"  max_abs_diff={diff.max():.2e}")
        print(f"  mean_abs_diff={diff.mean():.2e}")
        cos = np.dot(normed_embeds.ravel(), hf_data.ravel()) / (
            np.linalg.norm(normed_embeds.ravel()) * np.linalg.norm(hf_data.ravel())
        )
        print(f"  cosine={cos:.10f}")
        if diff.max() < 1e-5 and diff.mean() < 1e-6 and cos > 0.999999:
            print("  PASS")
        else:
            print("  FAIL")
            print(f"  First 5 values (ours):   {normed_embeds[0, :5]}")
            print(f"  First 5 values (HF):     {hf_data[0, :5]}")

    # Step 4: Single full layer check
    residual = token_embeds
    hidden_states = residual

    # Input layernorm
    normed = torch.tensor(normed_embeds)

    # QKV projection
    q_proj = layer0.self_attn.q_proj  # [2048, 1024]
    k_proj = layer0.self_attn.k_proj  # [1024, 1024]
    v_proj = layer0.self_attn.v_proj  # [1024, 1024]
    o_proj = layer0.self_attn.o_proj  # [1024, 2048]

    q = normed @ q_proj.weight.data.T  # [15, 2048]
    k = normed @ k_proj.weight.data.T  # [15, 1024]
    v = normed @ v_proj.weight.data.T  # [15, 1024]

    # Reshape for attention: [seq, num_heads, head_dim]
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    q = q.view(15, num_heads, head_dim)
    k = k.view(15, num_kv_heads, head_dim)
    v = v.view(15, num_kv_heads, head_dim)

    # QK norm
    q_norm = layer0.self_attn.q_norm
    k_norm = layer0.self_attn.k_norm
    q = q_norm(q)
    k = k_norm(k)

    # RoPE using HF's apply_rotary_pos_emb
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    position_ids = torch.arange(15).unsqueeze(0)
    cos, sin = model.model.rotary_emb(token_embeds.unsqueeze(0), position_ids)
    # cos, sin shape: [1, 15, 128]

    # Transpose for HF RoPE: [batch, heads, seq, dim]
    q_for_rope = q.unsqueeze(0).transpose(1, 2)  # [1, 16, 15, 128]
    k_for_rope = k.unsqueeze(0).transpose(1, 2)  # [1, 8, 15, 128]
    q_rotated, k_rotated = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin)
    # q_rotated: [1, 16, 15, 128], k_rotated: [1, 8, 15, 128]
    # Already in [heads, seq, dim] format after squeeze
    q_transposed = q_rotated.squeeze(0)  # [16, 15, 128]
    k_transposed = k_rotated.squeeze(0)  # [8, 15, 128]

    # Attention: Q @ K^T / sqrt(head_dim)
    scale = 1.0 / np.sqrt(head_dim)

    # GQA: repeat K/V heads
    n_groups = num_heads // num_kv_heads
    k_expanded = k_transposed.repeat_interleave(n_groups, dim=0)  # [16, 15, 128]
    v_transposed = v.transpose(0, 1)  # [8, 15, 128]
    v_expanded = v_transposed.repeat_interleave(n_groups, dim=0)  # [16, 15, 128]

    # QK^T
    attn_scores = torch.matmul(q_transposed, k_expanded.transpose(-2, -1)) * scale  # [16, 15, 15]

    # Causal mask
    mask = torch.triu(torch.ones(15, 15), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))

    # Softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # V @ probs
    attn_output = torch.matmul(attn_probs, v_expanded)  # [16, 15, 128]
    attn_output = attn_output.transpose(0, 1).contiguous().view(15, -1)  # [15, 2048]

    # O projection
    o_output = attn_output @ o_proj.weight.data.T  # [15, 1024]

    # Residual add
    attn_final = residual + o_output

    # Compare with HF attention output
    if (out_dir / "hf_layer00_attn_output.npy").exists():
        hf_data = np.load(str(out_dir / "hf_layer00_attn_output.npy"))
        our_data = o_output.detach().numpy()
        diff = np.abs(our_data - hf_data)
        print(f"\nAttention output: shape={our_data.shape}")
        print(f"  max_abs_diff={diff.max():.2e}")
        print(f"  mean_abs_diff={diff.mean():.2e}")
        # Per-token max error
        for tok in range(our_data.shape[0]):
            tok_diff = diff[tok].max()
            print(f"  token[{tok}] max_err={tok_diff:.2e}")
        cos = np.dot(our_data.ravel(), hf_data.ravel()) / (
            np.linalg.norm(our_data.ravel()) * np.linalg.norm(hf_data.ravel())
        )
        print(f"  cosine={cos:.10f}")
        if diff.max() < 1e-3:
            print("  PASS")
        else:
            print("  FAIL")

    # Post attention layernorm
    post_norm_weight = layer0.post_attention_layernorm.weight.data
    post_normed = torch.tensor(rms_norm(attn_final, post_norm_weight, eps))

    # MLP
    mlp = layer0.mlp
    gate = post_normed @ mlp.gate_proj.weight.data.T  # [15, 3072]
    up = post_normed @ mlp.up_proj.weight.data.T      # [15, 3072]
    intermediate = torch.nn.functional.silu(gate) * up  # [15, 3072]
    down = intermediate @ mlp.down_proj.weight.data.T  # [15, 1024]

    # Residual add
    mlp_final = attn_final + down

    # Compare with HF MLP output
    if (out_dir / "hf_layer00_mlp_output.npy").exists():
        hf_data = np.load(str(out_dir / "hf_layer00_mlp_output.npy"))
        our_data = down.detach().numpy()
        diff = np.abs(our_data - hf_data)
        print(f"\nMLP output (before residual): shape={our_data.shape}")
        print(f"  max_abs_diff={diff.max():.2e}")
        print(f"  mean_abs_diff={diff.mean():.2e}")
        cos = np.dot(our_data.ravel(), hf_data.ravel()) / (
            np.linalg.norm(our_data.ravel()) * np.linalg.norm(hf_data.ravel())
        )
        print(f"  cosine={cos:.10f}")

    # Compare with HF layer00_output (after residual)
    if (out_dir / "hf_layer00_output.npy").exists():
        hf_data = np.load(str(out_dir / "hf_layer00_output.npy"))
        our_data = mlp_final.detach().numpy()
        diff = np.abs(our_data - hf_data)
        print(f"\nLayer 0 output: shape={our_data.shape}")
        print(f"  max_abs_diff={diff.max():.2e}")
        print(f"  mean_abs_diff={diff.mean():.2e}")
        cos = np.dot(our_data.ravel(), hf_data.ravel()) / (
            np.linalg.norm(our_data.ravel()) * np.linalg.norm(hf_data.ravel())
        )
        print(f"  cosine={cos:.10f}")
        if diff.max() < 1e-3:
            print("  Layer 0 PASS")
        else:
            print("  Layer 0 FAIL")
            print(f"  First 5 values (ours):   {our_data[0, :5]}")
            print(f"  First 5 values (HF):     {hf_data[0, :5]}")


if __name__ == "__main__":
    main()
