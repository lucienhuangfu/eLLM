#!/usr/bin/env python3
"""Stream plain text from the local eLLM chat-completions endpoint."""

import argparse
import json
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream a response from eLLM")
    parser.add_argument("prompt", nargs="*", help="question; read interactively when omitted")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = input("You: ").strip()
    if not prompt:
        parser.error("prompt must not be empty")

    payload = json.dumps(
        {
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": max(args.max_tokens, 1),
        },
        ensure_ascii=False,
    )

    try:
        process = subprocess.Popen(
            [
                "curl",
                "-sS",
                "-N",
                args.url,
                "-H",
                "Content-Type: application/json",
                "-d",
                payload,
            ],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        print("curl was not found", file=sys.stderr)
        return 127

    assert process.stdout is not None
    print("Assistant: ", end="", flush=True)
    try:
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue

            event = json.loads(data)
            choice = event.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            text = delta.get("content") or delta.get("reasoning_content")
            if text:
                print(text, end="", flush=True)
    except (json.JSONDecodeError, KeyError, IndexError) as error:
        process.terminate()
        print(f"\nInvalid stream response: {error}", file=sys.stderr)
        return 1

    return_code = process.wait()
    print()
    if return_code != 0:
        print(f"curl exited with status {return_code}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
