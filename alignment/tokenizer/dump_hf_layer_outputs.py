#!/usr/bin/env python3
"""Dump HF Qwen3 intermediate activations for each decoder layer using hooks."""
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


def main() -> None:
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "alignment/tokenizer/dump")
    out_dir.mkdir(parents=True, exist_ok=True)

    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32,
    ).eval()
    model = model.to("cpu")

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]

    # Dump input_ids
    np.save(str(out_dir / "hf_input_ids.npy"), input_ids[0].numpy().astype(np.int32))

    # Register hooks to capture intermediate states
    hook_data = {}

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

    def make_mlp_output_hook(layer_idx):
        def hook(module, input, output):
            hook_data[f"layer{layer_idx:02d}_mlp_output"] = output[0].detach().numpy()
        return hook

    # Also hook the decoder layer input/output
    def make_layer_input_hook(layer_idx):
        def hook(module, input):
            hook_data[f"layer{layer_idx:02d}_input"] = input[0][0].detach().numpy()
        return hook

    def make_layer_output_hook(layer_idx):
        def hook(module, input, output):
            hook_data[f"layer{layer_idx:02d}_output"] = output[0].detach().numpy()
        return hook

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(make_layer_input_hook(layer_idx)))
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_input_layernorm_hook(layer_idx)))
        hooks.append(layer.self_attn.register_forward_hook(
            make_attn_output_hook(layer_idx)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_post_attn_norm_hook(layer_idx)))
        hooks.append(layer.mlp.register_forward_hook(
            make_mlp_output_hook(layer_idx)))
        hooks.append(layer.register_forward_hook(make_layer_output_hook(layer_idx)))

    # Hook final norm
    def final_norm_hook(module, input, output):
        hook_data["final_norm"] = output[0].detach().numpy()
    hooks.append(model.model.norm.register_forward_hook(final_norm_hook))

    # Hook embeddings
    def embed_hook(module, input, output):
        hook_data["embeddings"] = output[0].detach().numpy()
    hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        next_token_id = int(torch.argmax(logits).item())

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
        "next_token_id": next_token_id,
        "next_token": tokenizer.decode([next_token_id]),
    }, indent=2))


if __name__ == "__main__":
    main()
