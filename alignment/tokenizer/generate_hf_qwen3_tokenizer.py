#!/usr/bin/env python3
import json
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "third_party" / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    model_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B")
    messages = [{"role": "user", "content": "你好，请用一句话介绍 Rust。"}]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    token_ids = tokenized["input_ids"] if hasattr(tokenized, "__getitem__") else tokenized

    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "messages": messages,
                "add_generation_prompt": True,
                "rendered_prompt": prompt,
                "token_ids": token_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
