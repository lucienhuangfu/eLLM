#!/usr/bin/env python3
import json
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "third_party" / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main() -> None:
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    ).eval()

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        next_token_id = int(torch.argmax(logits).item())

    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "messages": messages,
                "rendered_prompt": prompt,
                "input_ids": inputs["input_ids"][0].tolist(),
                "next_token_id": next_token_id,
                "next_token": tokenizer.decode([next_token_id]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
