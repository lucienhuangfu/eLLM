#!/usr/bin/env python3
"""Detailed layer 0 sub-component comparison: Rust vs HF."""
import pathlib, sys
import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "third_party" / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import AutoModelForCausalLM, AutoTokenizer

DUMP_DIR = pathlib.Path("alignment/tokenizer/dump")
REF_DIR = pathlib.Path("alignment/matmul3/dump")


def rms_norm_f16(x, weight, eps):
    x_f16 = x.to(torch.float16)
    variance = x_f16.float().pow(2).mean(-1, keepdim=True)
    return (weight.to(torch.float16) * (x_f16.float() * torch.rsqrt(variance + eps)).to(torch.float16))


def rms_norm_unit_f16(head, eps):
    variance = head.float().pow(2).mean(-1, keepdim=True)
    return (head.float() * torch.rsqrt(variance + eps)).to(torch.float16)


def rotate_half_rope_f16(head, rope, head_dim):
    half = head_dim // 2
    head_f32 = head.float()
    rope_f32 = rope.float()
    rotated = torch.zeros_like(head_f32)
    for i in range(half):
        x1, x2 = head_f32[..., i], head_f32[..., i + half]
        cos, sin = rope_f32[..., 2 * i], rope_f32[..., 2 * i + 1]
        rotated[..., i] = x1 * cos - x2 * sin
        rotated[..., i + half] = x2 * cos + x1 * sin
    return rotated.to(torch.float16)


def read_f16_bin(path, shape):
    with open(path, "rb") as f:
        raw = f.read()
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float16).astype(np.float32)).reshape(shape)


def load_npy_tensor(path, shape):
    return torch.from_numpy(np.load(str(path))).reshape(shape)


def compare(name, rust_data, ref_data):
    diff = (rust_data.float() - ref_data.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos = torch.dot(rust_data.float().ravel(), ref_data.float().ravel()) / (
        torch.norm(rust_data.float().ravel()) * torch.norm(ref_data.float().ravel())
    )
    status = "PASS" if cos > 0.9999 else "FAIL"
    marker = "  *** FAIL ***" if status == "FAIL" else ""
    print(f"  {name:45s} max_err={max_err:.4e} mean_err={mean_err:.4e} cos={cos:.10f} {status}{marker}")
    if status == "FAIL":
        for tok in range(min(rust_data.shape[0], 15)):
            tok_diff = diff[tok].max().item()
            if tok_diff > 0.01:
                print(f"    token[{tok}]: max_err={tok_diff:.4e}")
    return max_err, cos, status


def main():
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False,
                                                  torch_dtype=torch.float16).eval()

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"][0]
    seq_len = len(input_ids)

    embed_weight = model.model.embed_tokens.weight.data
    eps = model.config.rms_norm_eps
    num_heads, num_kv_heads, head_dim, hidden_size = 16, 8, 128, 1024

    token_embeds = embed_weight[input_ids]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(token_embeds.unsqueeze(0), position_ids)

    # Build Rust-format rope table
    rope_rust = torch.zeros(seq_len, head_dim, dtype=torch.float16)
    for pos in range(seq_len):
        for i in range(head_dim // 2):
            rope_rust[pos, 2 * i] = cos[0, pos, i]
            rope_rust[pos, 2 * i + 1] = sin[0, pos, i]

    layer = model.model.layers[0]

    # === Compare with Rust dumps ===
    print("=" * 80)
    print("Step 0: Normed hidden (Input to MatMul3)")
    print("=" * 80)
    normed = rms_norm_f16(token_embeds, layer.input_layernorm.weight.data, eps)
    rust_normed = read_f16_bin(DUMP_DIR / "rust_layer0_normed_hidden.bin", [seq_len, hidden_size])
    compare("normed_hidden", rust_normed, normed)

    print("\n" + "=" * 80)
    print("Step 1: Q/K/V (MatMul3 output)")
    print("=" * 80)
    q_full = normed @ layer.self_attn.q_proj.weight.data.T
    k_full = normed @ layer.self_attn.k_proj.weight.data.T
    v_full = normed @ layer.self_attn.v_proj.weight.data.T

    q = q_full.view(seq_len, num_heads, head_dim)
    k = k_full.view(seq_len, num_kv_heads, head_dim)
    v = v_full.view(seq_len, num_kv_heads, head_dim)

    q_norm_w = layer.self_attn.q_norm.weight.data
    k_norm_w = layer.self_attn.k_norm.weight.data
    q_normed = rms_norm_unit_f16(q, eps)
    q = q_norm_w.to(torch.float16) * q_normed
    k_normed_h = rms_norm_unit_f16(k, eps)
    k = k_norm_w.to(torch.float16) * k_normed_h

    for tok in range(seq_len):
        rp = rope_rust[tok]
        for h in range(num_heads):
            q[tok, h] = rotate_half_rope_f16(q[tok, h], rp, head_dim)
        for h in range(num_kv_heads):
            k[tok, h] = rotate_half_rope_f16(k[tok, h], rp, head_dim)

    q_final = q.reshape(seq_len, num_heads * head_dim)
    k_final = k.reshape(seq_len, num_kv_heads * head_dim)

    rust_q = read_f16_bin(DUMP_DIR / "rust_layer0_q_output.bin", [seq_len, 2048])
    rust_k = read_f16_bin(DUMP_DIR / "rust_layer0_k_output.bin", [seq_len, 1024])
    rust_v = read_f16_bin(DUMP_DIR / "rust_layer0_v_output.bin", [seq_len, 1024])

    compare("Q_output", rust_q, q_final)
    compare("K_output", rust_k, k_final)
    compare("V_output", rust_v, v.reshape(seq_len, 1024))

    print("\n" + "=" * 80)
    print("Step 2: Attention (QK scores + softmax + V weighted sum)")
    print("=" * 80)
    scale = 1.0 / np.sqrt(head_dim)
    n_groups = num_heads // num_kv_heads

    q_attn = q.permute(1, 0, 2).unsqueeze(0)
    k_attn = k.permute(1, 0, 2).unsqueeze(0)
    v_attn = v.permute(1, 0, 2).unsqueeze(0)

    k_attn = k_attn.repeat_interleave(n_groups, dim=1)
    v_attn = v_attn.repeat_interleave(n_groups, dim=1)

    attn_scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(torch.float16)
    attn_output = torch.matmul(attn_probs, v_attn)
    attn_output = attn_output.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1)

    print(f"  attn_output shape: {attn_output.shape}")

    print("\n" + "=" * 80)
    print("Step 3: O projection (attention output @ W_o)")
    print("=" * 80)
    o_weight = layer.self_attn.o_proj.weight.data
    o_output = attn_output @ o_weight.T

    print(f"  o_output shape: {o_output.shape}")

    print("\n" + "=" * 80)
    print("Step 4: Residual add (attn_final = hidden + o_output)")
    print("=" * 80)
    residual = token_embeds
    attn_final = residual + o_output

    print("\n" + "=" * 80)
    print("Step 5: Post-attention RMSNorm")
    print("=" * 80)
    post_normed = rms_norm_f16(attn_final, layer.post_attention_layernorm.weight.data, eps)

    print("\n" + "=" * 80)
    print("Step 6: MLP gate/up projections + SiLU multiply")
    print("=" * 80)
    gate = post_normed @ layer.mlp.gate_proj.weight.data.T
    up = post_normed @ layer.mlp.up_proj.weight.data.T
    intermediate = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.float16)

    print("\n" + "=" * 80)
    print("Step 7: MLP down projection + residual (layer output)")
    print("=" * 80)
    down = intermediate @ layer.mlp.down_proj.weight.data.T
    layer_output = attn_final + down

    # Compare final layer output
    print("\n" + "=" * 80)
    print("FINAL: Layer 0 output vs HF reference")
    print("=" * 80)
    hf_layer0 = load_npy_tensor(DUMP_DIR / "hf_layer00_output.npy", [seq_len, hidden_size])
    compare("layer0_output vs HF f32 ref", layer_output, hf_layer0)

    # Also compare sub-components
    print("\n" + "=" * 80)
    print("Sub-component comparison vs HF reference")
    print("=" * 80)
    hf_attn = load_npy_tensor(DUMP_DIR / "hf_layer00_attn_output.npy", [seq_len, hidden_size])
    hf_mlp = load_npy_tensor(DUMP_DIR / "hf_layer00_mlp_output.npy", [seq_len, hidden_size])
    hf_post_norm = load_npy_tensor(DUMP_DIR / "hf_layer00_post_attn_norm.npy", [seq_len, hidden_size])

    compare("O_output vs hf_attn_output", o_output, hf_attn)
    compare("post_attn_norm vs hf_post_attn_norm", post_normed, hf_post_norm)
    compare("MLP_down vs hf_mlp_output", down, hf_mlp)

    # Finally, compare Rust's layer output directly
    print("\n" + "=" * 80)
    print("Rust layer0 vs Python mirror layer0")
    print("=" * 80)
    rust_layer0 = read_f16_bin(DUMP_DIR / "rust_layer00_output.bin", [seq_len, hidden_size])
    compare("Rust layer0 vs Python mirror", rust_layer0, layer_output)
    compare("Rust layer0 vs HF f32 ref", rust_layer0, hf_layer0)


if __name__ == "__main__":
    main()
