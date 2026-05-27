#!/usr/bin/env python3
"""Mirror Rust operator sequence in f16 to find divergence point."""
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "third_party" / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


def rms_norm_f16(x, weight, eps):
    """RMSNorm in f16 (as Rust does it)."""
    x_f16 = x.to(torch.float16)
    variance = x_f16.float().pow(2).mean(-1, keepdim=True)
    x_normed = (x_f16.float() * torch.rsqrt(variance + eps)).to(torch.float16)
    return (weight.to(torch.float16) * x_normed)


def rms_norm_unit_f16(head, eps):
    """RMSNorm without weight, in f16."""
    variance = head.float().pow(2).mean(-1, keepdim=True)
    return (head.float() * torch.rsqrt(variance + eps)).to(torch.float16)


def rotate_half_rope_f16(head, rope, head_dim):
    """Mirror Rust rotate_half_rope in f16."""
    half = head_dim // 2
    # rope format: [cos0, sin0, cos1, sin1, ...]
    head_f32 = head.float()
    rope_f32 = rope.float()
    rotated = torch.zeros_like(head_f32)
    for i in range(half):
        x1 = head_f32[..., i]
        x2 = head_f32[..., i + half]
        cos = rope_f32[..., 2 * i]
        sin = rope_f32[..., 2 * i + 1]
        rotated[..., i] = x1 * cos - x2 * sin
        rotated[..., i + half] = x2 * cos + x1 * sin
    return rotated.to(torch.float16)


def main() -> None:
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    out_dir = pathlib.Path("alignment/tokenizer/dump")
    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    # Load in f16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16,
    ).eval()

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"][0]
    seq_len = len(input_ids)

    embed_weight = model.model.embed_tokens.weight.data
    eps = model.config.rms_norm_eps
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    hidden_size = 1024

    # Get f16 embeddings
    token_embeds = embed_weight[input_ids]  # f16 since model is f16

    # Compute RoPE table (Rust format: [cos0, sin0, cos1, sin1, ...] per position)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(token_embeds.unsqueeze(0), position_ids)
    # HF cos/sin: [1, seq_len, 128] with paired format [c0,c0,c1,c1,...,s0,s0,s1,s1,...]
    # Convert to Rust interleaved format [c0,s0,c1,s1,...] per position
    rope_rust = torch.zeros(seq_len, head_dim, dtype=torch.float16)
    for pos in range(seq_len):
        for i in range(head_dim // 2):
            # HF cos/sin format: [c0,c1,...,c63,c0,c1,...,c63]
            # Rust needs interleaved [c0,s0,c1,s1,...,c63,s63]
            rope_rust[pos, 2 * i] = cos[0, pos, i]
            rope_rust[pos, 2 * i + 1] = sin[0, pos, i]

    # ===== Manual computation, mirroring Rust operator sequence =====
    # Layer 0: LookupRMSMap (embedding lookup + RMSNorm in one op)
    # Rust does: lookup embedding, then RMSNorm
    hidden = token_embeds.clone()  # [15, 1024]

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden

        # 1. RMSMap (or LookupRMSMap for layer 0)
        normed = rms_norm_f16(hidden, layer.input_layernorm.weight.data, eps)

        # 2. MatMul3 (QKV + QK Norm + RoPE, GEMV path)
        q_weight = layer.self_attn.q_proj.weight.data  # [2048, 1024] f16
        k_weight = layer.self_attn.k_proj.weight.data  # [1024, 1024]
        v_weight = layer.self_attn.v_proj.weight.data  # [1024, 1024]
        q_norm_w = layer.self_attn.q_norm.weight.data  # [128]
        k_norm_w = layer.self_attn.k_norm.weight.data  # [128]

        # Rust computes Q, K, V per-head using GEMV
        # Q: [seq_len, num_heads * head_dim] = [15, 2048]
        # K: [seq_len, num_kv_heads * head_dim] = [15, 1024]
        # V: [seq_len, num_kv_heads * head_dim] = [15, 1024]

        # Do full matmul in f16 (same as Rust GEMV per-head)
        q_full = (normed @ q_weight.T)  # [15, 2048]
        k_full = (normed @ k_weight.T)  # [15, 1024]
        v_full = (normed @ v_weight.T)  # [15, 1024]

        # Reshape to heads
        q = q_full.view(seq_len, num_heads, head_dim)  # [15, 16, 128]
        k = k_full.view(seq_len, num_kv_heads, head_dim)  # [15, 8, 128]
        v = v_full.view(seq_len, num_kv_heads, head_dim)  # [15, 8, 128]

        # QK Norm (RMSNorm without weight, then apply weight) - mirror Rust compute_norm_rope
        # For Q:
        q_normed = rms_norm_unit_f16(q, eps)
        q = q_norm_w.to(torch.float16) * q_normed
        # For K:
        k_normed = rms_norm_unit_f16(k, eps)
        k = k_norm_w.to(torch.float16) * k_normed

        # RoPE (Rust rotate_half_rope, per-token, per-head)
        for tok in range(seq_len):
            rope_per_pos = rope_rust[tok]  # [128]
            for h in range(num_heads):
                q[tok, h] = rotate_half_rope_f16(q[tok, h], rope_per_pos, head_dim)
            for h in range(num_kv_heads):
                k[tok, h] = rotate_half_rope_f16(k[tok, h], rope_per_pos, head_dim)

        # 3. Attention
        scale = 1.0 / np.sqrt(head_dim)
        n_groups = num_heads // num_kv_heads

        # Reshape for attention: Q [seq, heads, dim], K [seq, kv_heads, dim], V [seq, kv_heads, dim]
        # HF format: [batch, heads, seq, dim]
        q_attn = q.permute(1, 0, 2).unsqueeze(0)  # [1, 16, 15, 128]
        k_attn = k.permute(1, 0, 2).unsqueeze(0)  # [1, 8, 15, 128]
        v_attn = v.permute(1, 0, 2).unsqueeze(0)  # [1, 8, 15, 128]

        # GQA: repeat K/V
        k_attn = k_attn.repeat_interleave(n_groups, dim=1)  # [1, 16, 15, 128]
        v_attn = v_attn.repeat_interleave(n_groups, dim=1)

        # Attention scores
        attn_scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(torch.float16)
        attn_output = torch.matmul(attn_probs, v_attn)  # [1, 16, 15, 128]
        attn_output = attn_output.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1)  # [15, 2048]

        # 4. O projection + residual (MatMulAdd in Rust)
        o_weight = layer.self_attn.o_proj.weight.data  # [1024, 2048]
        o_output = attn_output @ o_weight.T  # [15, 1024]
        attn_final = residual + o_output

        # 5. Post-attention RMSNorm
        post_normed = rms_norm_f16(attn_final, layer.post_attention_layernorm.weight.data, eps)

        # 6. MLP: gate projection
        gate = post_normed @ layer.mlp.gate_proj.weight.data.T  # [15, 3072]
        up = post_normed @ layer.mlp.up_proj.weight.data.T  # [15, 3072]

        # 7. SiLU + multiply (SiluMulZipMap in Rust)
        intermediate = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.float16)

        # 8. Down projection + residual (MatMulAdd in Rust)
        down = intermediate @ layer.mlp.down_proj.weight.data.T  # [15, 1024]
        hidden = attn_final + down

        # Compare with HF f32 dump
        hf_dump = out_dir / f"hf_layer{layer_idx:02d}_output.npy"
        if hf_dump.exists():
            hf_data = torch.from_numpy(np.load(str(hf_dump)))
            our = hidden.float()
            diff = (our - hf_data).abs()
            cos = torch.dot(our.ravel(), hf_data.ravel()) / (
                torch.norm(our.ravel()) * torch.norm(hf_data.ravel())
            )
            print(f"Layer {layer_idx:02d}: max_err={diff.max():.4e} mean_err={diff.mean():.4e} cos={cos:.10f}")
            if cos < 0.99:
                print(f"  *** DIVERGING at layer {layer_idx} ***")
                # Check sub-components
                # Compare attention output
                hf_attn = torch.from_numpy(np.load(str(out_dir / f"hf_layer{layer_idx:02d}_attn_output.npy")))
                attn_diff = (o_output.float() - hf_attn).abs()
                print(f"    attn_output max_err={attn_diff.max():.4e}")
                # Compare MLP output
                hf_mlp = torch.from_numpy(np.load(str(out_dir / f"hf_layer{layer_idx:02d}_mlp_output.npy")))
                mlp_diff = (down.float() - hf_mlp).abs()
                print(f"    mlp_output max_err={mlp_diff.max():.4e}")
                # Compare post_norm
                hf_post_norm = torch.from_numpy(np.load(str(out_dir / f"hf_layer{layer_idx:02d}_post_attn_norm.npy")))
                pn_diff = (post_normed.float() - hf_post_norm).abs()
                print(f"    post_norm max_err={pn_diff.max():.4e}")

    # Final norm
    final_norm = rms_norm_f16(hidden, model.model.norm.weight.data, eps)

    # LM head
    lm_head = model.lm_head.weight.data  # [151936, 1024]
    logits = (final_norm @ lm_head.T).float()
    last_logits = logits[-1]
    top_token = int(torch.argmax(last_logits).item())

    # Also compute with HF model directly
    with torch.no_grad():
        hf_out = model(**inputs)
        hf_top = int(torch.argmax(hf_out.logits[0, -1, :].float()).item())

    print(f"\nManual f16: top token = {top_token} ({tokenizer.decode([top_token])!r})")
    print(f"HF f16:     top token = {hf_top} ({tokenizer.decode([hf_top])!r})")
    print(f"HF f32 ref: top token = 151667 (<think>)")


if __name__ == "__main__":
    main()
