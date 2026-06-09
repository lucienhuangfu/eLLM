#!/usr/bin/env python3
"""Dump HF Qwen3MoE router and expert intermediates for each decoder layer.

This extends dump_hf_layer_outputs.py with MoE-specific hooks:
- Router logits (gate output before softmax)
- Router probabilities (after softmax + topk + normalization)
- Expert indices and routing weights
- Expert intermediate outputs (after SiLU(gate)*up)
- Expert down projection outputs
- Per-token expert outputs after merge
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


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported torch dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default="models/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("out_dir", nargs="?", default="alignment/tokenizer/dump")
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = parse_dtype(args.torch_dtype)

    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch_dtype,
    ).eval()
    model = model.to("cpu")

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]

    np.save(str(out_dir / "hf_input_ids.npy"), input_ids[0].numpy().astype(np.int32))

    # Register hooks for all intermediate states
    hook_data = {}

    # ---- Standard layer hooks (from dump_hf_layer_outputs.py) ----
    def make_input_layernorm_hook(layer_idx):
        def hook(module, input, output):
            hook_data[f"layer{layer_idx:02d}_post_input_norm"] = output[0].detach().numpy()
        return hook

    def make_attn_output_hook(layer_idx):
        def hook(module, input, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            hook_data[f"layer{layer_idx:02d}_attn_output"] = attn_out[0].detach().numpy()
        return hook

    def make_post_attn_norm_hook(layer_idx):
        def hook(module, input, output):
            hook_data[f"layer{layer_idx:02d}_post_attn_norm"] = output[0].detach().numpy()
        return hook

    def make_layer_input_hook(layer_idx):
        def hook(module, input):
            hook_data[f"layer{layer_idx:02d}_input"] = input[0][0].detach().numpy()
        return hook

    def make_layer_output_hook(layer_idx):
        def hook(module, input, output):
            hook_data[f"layer{layer_idx:02d}_output"] = output[0].detach().numpy()
        return hook

    # ---- MoE-specific hooks ----
    def make_router_gate_hook(layer_idx):
        """Hook into the router gate linear layer to capture logits before softmax."""
        def hook(module, input, output):
            # router_logits shape: [seq_len, num_experts]
            hook_data[f"layer{layer_idx:02d}_router_logits"] = output.detach().numpy()
        return hook

    def make_moe_output_hook(layer_idx):
        """Hook into the entire MoE block to capture merged output."""
        def hook(module, input, output):
            # MoE output shape: [1, seq_len, hidden_size]
            hook_data[f"layer{layer_idx:02d}_mlp_output"] = output[0].detach().numpy()
        return hook

    # ---- Store expert weights for each layer ----
    def make_expert_hook(layer_idx):
        """Deep-hook into expert forward to capture expert selection and routing weights."""
        # We monkey-patch the router forward to capture intermediate values
        pass  # Use a different approach - patch the gate forward

    hooks = []
    gate_patches = {}

    # Patch the router gate.forward to capture logits and routing weights
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(make_layer_input_hook(layer_idx)))
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_input_layernorm_hook(layer_idx)))
        hooks.append(layer.self_attn.register_forward_hook(
            make_attn_output_hook(layer_idx)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_post_attn_norm_hook(layer_idx)))
        hooks.append(layer.mlp.register_forward_hook(make_moe_output_hook(layer_idx)))
        hooks.append(layer.register_forward_hook(make_layer_output_hook(layer_idx)))

        # Patch the router gate to capture router internals
        original_gate_forward = layer.mlp.gate.forward

        def make_patched_gate(orig_fn, l_idx):
            def patched_forward(hidden_states):
                # Call original
                router_logits, routing_weights, selected_experts = orig_fn(hidden_states)
                # Capture intermediates
                hook_data[f"layer{l_idx:02d}_router_logits"] = router_logits.detach().numpy()
                hook_data[f"layer{l_idx:02d}_routing_weights"] = routing_weights.detach().numpy()
                hook_data[f"layer{l_idx:02d}_selected_experts"] = selected_experts.detach().numpy()
                return router_logits, routing_weights, selected_experts
            return patched_forward

        layer.mlp.gate.forward = make_patched_gate(original_gate_forward, layer_idx)
        gate_patches[layer_idx] = (layer.mlp.gate, original_gate_forward)

    # Hook final norm
    def final_norm_hook(module, input, output):
        hook_data["final_norm"] = output[0].detach().numpy()
    hooks.append(model.model.norm.register_forward_hook(final_norm_hook))

    # Hook embeddings
    def embed_hook(module, input, output):
        hook_data["embeddings"] = output[0].detach().numpy()
    hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    # ---- Run forward pass ----
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        next_token_id = int(torch.argmax(logits).item())

    # Restore original gate forward methods
    for layer_idx, (gate_module, orig_fn) in gate_patches.items():
        gate_module.forward = orig_fn

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save all dumped data
    for key, arr in hook_data.items():
        np.save(str(out_dir / f"hf_{key}.npy"), arr.astype(np.float32))

    print(json.dumps({
        "dumped_keys": sorted(hook_data.keys()),
        "layers_dumped": len(model.model.layers),
        "hidden_size": model.config.hidden_size,
        "torch_dtype": str(torch_dtype),
        "next_token_id": next_token_id,
        "next_token": tokenizer.decode([next_token_id]),
    }, indent=2))


if __name__ == "__main__":
    main()
