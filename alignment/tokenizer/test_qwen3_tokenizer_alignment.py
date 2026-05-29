#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_json_from_output(output: str) -> dict:
    start = output.rfind("\n{")
    if start >= 0:
        start += 1
    else:
        start = output.find("{")
    if start < 0:
        raise ValueError(f"no JSON object found in output:\n{output}")
    return json.loads(output[start:])


def run(cmd: list[str]) -> dict:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return load_json_from_output(result.stdout)


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B"
    rust = run(["cargo", "run", "--quiet", "--bin", "qwen3_tokenizer_alignment", "--", model_dir])
    hf = run(["python3", "alignment/tokenizer/generate_hf_qwen3_tokenizer.py", model_dir])

    if rust["rendered_prompt"] != hf["rendered_prompt"]:
        raise AssertionError(
            "rendered prompt mismatch\n"
            f"rust={rust['rendered_prompt']!r}\n"
            f"hf={hf['rendered_prompt']!r}"
        )
    if rust["token_ids"] != hf["token_ids"]:
        raise AssertionError(
            "token ids mismatch\n"
            f"rust={rust['token_ids']}\n"
            f"hf={hf['token_ids']}"
        )

    print(json.dumps({"status": "ok", "token_count": len(rust["token_ids"])}, indent=2))


if __name__ == "__main__":
    main()
