#!/usr/bin/env python3
"""Generate test data for MatMul3 alignment: hidden states + weights + expected Q/K/V outputs.

Mirrors the Rust GEMV path: for each (row, head), compute dot product,
then apply RMSNorm + RoPE for Q and K heads.
"""
import pathlib, sys, struct
import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "third_party" / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import AutoModelForCausalLM, AutoTokenizer


def rms_norm_f16(x, weight, eps):
    """RMSNorm in f16 - mirrors Rust rms_norm kernel."""
    x_f16 = x.to(torch.float16)
    w_f16 = weight.to(torch.float16)
    # Accumulate in f32 like Rust
    variance = x_f16.float().pow(2).mean(-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    return (w_f16.float() * x_f16.float() * rrms).to(torch.float16)


def rms_norm_unit_f16(x, eps):
    """RMSNorm without weight in f16."""
    x_f16 = x.to(torch.float16)
    variance = x_f16.float().pow(2).mean(-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    return (x_f16.float() * rrms).to(torch.float16)


def rotate_half_rope_f16(head, rope, head_dim):
    """Mirror Rust rotate_half_rope in f16. rope is [cos0, sin0, cos1, sin1, ...]"""
    half = head_dim // 2
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


def save_f32(path, tensor):
    """Save as float32 1D npy (compatible with Rust npy crate)."""
    np.save(str(path), tensor.detach().cpu().to(torch.float32).numpy().ravel())


def save_f16_bin(path, tensor):
    """Save raw f16 bytes."""
    data = tensor.detach().cpu().to(torch.float16).numpy().ravel()
    with open(path, "wb") as f:
        f.write(data.tobytes())


def main():
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    out_dir = pathlib.Path("alignment/matmul3/dump")
    out_dir.mkdir(parents=True, exist_ok=True)

    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
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

    # Step 1: Embedding lookup + RMSNorm (LookupRMSMap in Rust)
    layer0 = model.model.layers[0]
    token_embeds = embed_weight[input_ids]  # [15, 1024] f16

    # RMSNorm (what LookupRMSMap does)
    normed_hidden = rms_norm_f16(token_embeds, layer0.input_layernorm.weight.data, eps)

    # Save the hidden states (input to MatMul3)
    save_f32(out_dir / "hidden_states.npy", normed_hidden)
    save_f16_bin(out_dir / "hidden_states_f16.bin", normed_hidden)

    # Step 2: Q, K, V weights
    q_weight = layer0.self_attn.q_proj.weight.data  # [2048, 1024]
    k_weight = layer0.self_attn.k_proj.weight.data  # [1024, 1024]
    v_weight = layer0.self_attn.v_proj.weight.data  # [1024, 1024]
    q_norm_w = layer0.self_attn.q_norm.weight.data  # [128]
    k_norm_w = layer0.self_attn.k_norm.weight.data  # [128]

    save_f32(out_dir / "q_weight.npy", q_weight)
    save_f32(out_dir / "k_weight.npy", k_weight)
    save_f32(out_dir / "v_weight.npy", v_weight)
    save_f32(out_dir / "q_norm_weight.npy", q_norm_w)
    save_f32(out_dir / "k_norm_weight.npy", k_norm_w)

    # Step 3: Compute RoPE table [seq_len, head_dim] in Rust interleaved format
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(token_embeds.unsqueeze(0), position_ids)
    rope_rust = torch.zeros(seq_len, head_dim, dtype=torch.float16)
    for pos in range(seq_len):
        for i in range(head_dim // 2):
            # HF cos/sin format: [c0,c1,...,c63,c0,c1,...,c63] (128 values)
            # We need interleaved [c0,s0,c1,s1,...,c63,s63]
            rope_rust[pos, 2 * i] = cos[0, pos, i]
            rope_rust[pos, 2 * i + 1] = sin[0, pos, i]
    save_f32(out_dir / "rope_table.npy", rope_rust)

    # Step 4: Compute Q, K, V per-head using GEMV (mirroring Rust compute_head_from_packed)
    # This is the same computation that mirror_rust_ops.py does
    hs_f16 = normed_hidden.to(torch.float16)

    # Full matmul in f16
    q_full = (hs_f16 @ q_weight.to(torch.float16).T).to(torch.float32)  # [15, 2048]
    k_full = (hs_f16 @ k_weight.to(torch.float16).T).to(torch.float32)  # [15, 1024]
    v_full = (hs_f16 @ v_weight.to(torch.float16).T).to(torch.float32)  # [15, 1024]

    save_f32(out_dir / "q_full_before_norm.npy", q_full)
    save_f32(out_dir / "k_full_before_norm.npy", k_full)
    save_f32(out_dir / "v_full.npy", v_full)

    # Reshape to heads
    q = q_full.view(seq_len, num_heads, head_dim)  # [15, 16, 128]
    k = k_full.view(seq_len, num_kv_heads, head_dim)  # [15, 8, 128]
    v = v_full.view(seq_len, num_kv_heads, head_dim)  # [15, 8, 128]

    # QK Norm: RMSNorm unit + weight multiply
    q_normed = rms_norm_unit_f16(q, eps)
    q = (q_norm_w.to(torch.float16) * q_normed).to(torch.float32)
    k_normed = rms_norm_unit_f16(k, eps)
    k = (k_norm_w.to(torch.float16) * k_normed).to(torch.float32)

    save_f32(out_dir / "q_after_norm.npy", q)
    save_f32(out_dir / "k_after_norm.npy", k)

    # RoPE
    for tok in range(seq_len):
        rope_per_pos = rope_rust[tok]  # [128]
        for h in range(num_heads):
            q[tok, h] = rotate_half_rope_f16(
                q[tok, h].to(torch.float16), rope_per_pos, head_dim
            ).to(torch.float32)
        for h in range(num_kv_heads):
            k[tok, h] = rotate_half_rope_f16(
                k[tok, h].to(torch.float16), rope_per_pos, head_dim
            ).to(torch.float32)

    # Flatten back to 2D
    q_final = q.reshape(seq_len, num_heads * head_dim)  # [15, 2048]
    k_final = k.reshape(seq_len, num_kv_heads * head_dim)  # [15, 1024]

    save_f32(out_dir / "q_final.npy", q_final)
    save_f32(out_dir / "k_final.npy", k_final)
    save_f32(out_dir / "v_final.npy", v_full)

    print(f"Saved test data to {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")

    # Print shapes for reference
    print(f"\nShapes:")
    print(f"  hidden_states: {list(normed_hidden.shape)}")
    print(f"  q_weight: {list(q_weight.shape)}")
    print(f"  k_weight: {list(k_weight.shape)}")
    print(f"  v_weight: {list(v_weight.shape)}")
    print(f"  q_final: {list(q_final.shape)}")
    print(f"  k_final: {list(k_final.shape)}")
    print(f"  v_final: {list(v_full.shape)}")
    print(f"  rope_table: {list(rope_rust.shape)}")


if __name__ == "__main__":
    main()
