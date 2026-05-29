#!/usr/bin/env python3
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
DUMP = ROOT / "alignment" / "tokenizer" / "dump"


def main() -> None:
    ellm = json.loads((DUMP / "ellm_qwen3_06b_tokenizer.json").read_text())
    hf = json.loads((DUMP / "hf_qwen3_06b_tokenizer.json").read_text())

    prompt_equal = ellm["rendered_prompt"] == hf["rendered_prompt"]
    ids_equal = ellm["token_ids"] == hf["token_ids"]

    print(f"prompt_equal={prompt_equal}")
    print(f"ids_equal={ids_equal}")
    print(f"token_count={len(ellm['token_ids'])}")

    if not prompt_equal:
        raise SystemExit("rendered prompt mismatch")
    if not ids_equal:
        raise SystemExit("token ids mismatch")


if __name__ == "__main__":
    main()
