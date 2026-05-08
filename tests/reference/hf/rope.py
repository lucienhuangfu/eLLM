#!/usr/bin/env python3
"""Hugging Face style RoPE reference implementation.

This script is intentionally dependency-free so it can be used as a small
oracle for Rust alignment tests.

Input format:
{
  "head_dim": 8,
  "rotary_dim": 4,
  "max_sequence_length": 2,
  "theta": 10000.0,
  "rope_scaling": {... optional ...}
}

Output format:
{
  "head_dim": 8,
  "rotary_dim": 4,
  "max_sequence_length": 2,
  "theta": 10000.0,
  "attention_scaling": 1.0,
  "values": [...]
}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def value_to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def value_to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def inv_freqs(dim: int, theta: float) -> list[float]:
    if dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    return [theta ** (-(i / dim)) for i in range(0, dim, 2)]


def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2.0 * math.pi))
        / (2.0 * math.log(base))
    )


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    low = max(0, int(low))
    high = min(int(high), max(dim - 1, 0))
    return low, high


def yarn_linear_ramp_mask(low: float, high: float, dim: int) -> list[float]:
    if abs(low - high) < 1e-12:
        high += 0.001

    out: list[float] = []
    for i in range(dim):
        linear = (i - low) / (high - low)
        out.append(min(1.0, max(0.0, linear)))
    return out


def parse_rope_scaling(
    rope_scaling: dict[str, Any] | None,
    default_original_max_position_embeddings: int,
) -> dict[str, float | int] | None:
    if not rope_scaling:
        return None

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if not isinstance(rope_type, str) or rope_type.lower() != "yarn":
        return None

    factor = value_to_float(rope_scaling.get("factor"))
    if factor is None or factor <= 0.0:
        factor = 1.0

    original_max_position_embeddings = value_to_int(
        rope_scaling.get("original_max_position_embeddings")
    )
    if original_max_position_embeddings is None:
        original_max_position_embeddings = default_original_max_position_embeddings

    beta_fast = value_to_float(rope_scaling.get("beta_fast"))
    if beta_fast is None:
        beta_fast = 32.0

    beta_slow = value_to_float(rope_scaling.get("beta_slow"))
    if beta_slow is None:
        beta_slow = 1.0

    attention_factor = value_to_float(rope_scaling.get("attention_factor"))
    if attention_factor is None:
        attention_factor = value_to_float(rope_scaling.get("attn_factor"))
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0

    return {
        "factor": factor,
        "original_max_position_embeddings": original_max_position_embeddings,
        "attention_factor": attention_factor,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
    }


def apply_yarn_scaling(
    inv_freqs_values: list[float],
    rotary_dim: int,
    theta: float,
    yarn: dict[str, float | int],
) -> None:
    factor = float(yarn["factor"])
    if factor <= 1.0:
        return

    rotary_pairs = rotary_dim // 2
    if rotary_pairs == 0:
        return

    # Rust's `round()` rounds half-away-from-zero. These values are positive,
    # so `floor(x + 0.5)` matches the same result.
    low_rot = max(1, int(math.floor(float(yarn["beta_fast"]) + 0.5)))
    high_rot = max(1, int(math.floor(float(yarn["beta_slow"]) + 0.5)))
    low, high = yarn_find_correction_range(
        low_rot,
        high_rot,
        rotary_dim,
        theta,
        int(yarn["original_max_position_embeddings"]),
    )
    ramp = yarn_linear_ramp_mask(float(low), float(high), rotary_pairs)
    inv_freq_extrapolation = list(inv_freqs_values)
    inv_freq_interpolation = [freq / factor for freq in inv_freq_extrapolation]

    for i in range(rotary_pairs):
        inv_freq_mask = 1.0 - ramp[i]
        inv_freqs_values[i] = (
            inv_freq_interpolation[i] * (1.0 - inv_freq_mask)
            + inv_freq_extrapolation[i] * inv_freq_mask
        )


def generate_rotary_embedding(case: dict[str, Any]) -> dict[str, Any]:
    head_dim = int(case["head_dim"])
    rotary_dim = min(int(case["rotary_dim"]), head_dim)
    max_sequence_length = int(case["max_sequence_length"])
    theta = float(case["theta"])

    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    if rotary_dim % 2 != 0:
        raise ValueError("RoPE rotary_dim must be even")

    yarn = parse_rope_scaling(case.get("rope_scaling"), max_sequence_length)
    attention_scaling = float(yarn["attention_factor"]) if yarn else 1.0

    inv = inv_freqs(rotary_dim, theta)
    if yarn is not None:
        apply_yarn_scaling(inv, rotary_dim, theta, yarn)

    values: list[float] = []
    rotary_pairs = rotary_dim // 2
    tail_pairs = head_dim // 2 - rotary_pairs

    for pos in range(max_sequence_length):
        t = float(pos)

        for inv_f in inv:
            angle = t * inv_f
            values.append(math.cos(angle) * attention_scaling)
            values.append(math.sin(angle) * attention_scaling)

        for _ in range(tail_pairs):
            values.append(1.0)
            values.append(0.0)

    return {
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "max_sequence_length": max_sequence_length,
        "theta": theta,
        "attention_scaling": attention_scaling,
        "values": values,
    }


def load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        case = json.load(f)
    if not isinstance(case, dict):
        raise ValueError("case file must contain a JSON object")
    return case


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RoPE reference data.")
    parser.add_argument("--case", required=True, type=Path, help="Input case JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file. If omitted, write to stdout.",
    )
    args = parser.parse_args()

    case = load_case(args.case)
    result = generate_rotary_embedding(case)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)

    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
