#!/usr/bin/env python3
"""Mirror Rust ops, save f16 layer outputs for comparison."""
import pathlib, sys
import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "third_party" / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import AutoModelForCausalLM, AutoTokenizer


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
    print(f"  {name:40s} max_err={max_err:.4e} mean_err={mean_err:.4e} cos={cos:.10f} {status}{marker}")
    return max_err, cos, status


def main():
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    out_dir = pathlib.Path("alignment/tokenizer/dump")
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

    rope_rust = torch.zeros(seq_len, head_dim, dtype=torch.float16)
    for pos in range(seq_len):
        for i in range(head_dim // 2):
            rope_rust[pos, 2 * i] = cos[0, pos, i]
            rope_rust[pos, 2 * i + 1] = sin[0, pos, i]

    hidden = token_embeds.clone()

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        normed = rms_norm_f16(hidden, layer.input_layernorm.weight.data, eps)

        q_full = normed @ layer.self_attn.q_proj.weight.data.T
        k_full = normed @ layer.self_attn.k_proj.weight.data.T
        v_full = normed @ layer.self_attn.v_proj.weight.data.T

        q = q_full.view(seq_len, num_heads, head_dim)
        k = k_full.view(seq_len, num_kv_heads, head_dim)
        v = v_full.view(seq_len, num_kv_heads, head_dim)

        q_norm_w = layer.self_attn.q_norm.weight.data
        k_norm_w = layer.self_attn.k_norm.weight.data
        q = q_norm_w.to(torch.float16) * rms_norm_unit_f16(q, eps)
        k = k_norm_w.to(torch.float16) * rms_norm_unit_f16(k, eps)

        for tok in range(seq_len):
            rp = rope_rust[tok]
            for h in range(num_heads):
                q[tok, h] = rotate_half_rope_f16(q[tok, h], rp, head_dim)
            for h in range(num_kv_heads):
                k[tok, h] = rotate_half_rope_f16(k[tok, h], rp, head_dim)

        # Attention
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
        attn_out = torch.matmul(attn_probs, v_attn)
        attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1)

        o_weight = layer.self_attn.o_proj.weight.data
        o_output = attn_out @ o_weight.T
        attn_final = residual + o_output

        post_normed = rms_norm_f16(attn_final, layer.post_attention_layernorm.weight.data, eps)

        gate = post_normed @ layer.mlp.gate_proj.weight.data.T
        up = post_normed @ layer.mlp.up_proj.weight.data.T
        intermediate = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.float16)

        down = intermediate @ layer.mlp.down_proj.weight.data.T
        hidden = attn_final + down

        # Save f16 layer output as raw binary (same format as Rust dump)
        data = hidden.detach().cpu().to(torch.float16).numpy().ravel()
        with open(out_dir / f"py_layer{layer_idx:02d}_output.bin", "wb") as f:
            f.write(data.tobytes())

        print(f"Layer {layer_idx:02d}: saved, shape={list(hidden.shape)}, "
              f"range=[{hidden.min().item():.4f}, {hidden.max().item():.4f}]")

    # Now compare Rust dumps against Python mirror
    dump_dir = pathlib.Path("alignment/tokenizer/dump")
    print("\n" + "=" * 80)
    print("Rust vs Python mirror comparison")
    print("=" * 80)
    for layer_idx in range(28):
        rust_path = dump_dir / f"rust_layer{layer_idx:02d}_output.bin"
        py_path = dump_dir / f"py_layer{layer_idx:02d}_output.bin"
        if not rust_path.exists():
            print(f"  layer {layer_idx:2d}: missing rust dump")
            continue
        rust_data = read_f16_bin(rust_path, [seq_len, hidden_size])
        py_data = read_f16_bin(py_path, [seq_len, hidden_size])
        compare(f"layer_{layer_idx:02d}", rust_data, py_data)


if __name__ == "__main__":
    main()
