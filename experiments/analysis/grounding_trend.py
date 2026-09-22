"""Cochran--Armitage trend test for the reported GS quartiles."""
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUARTILES = [
    {"name": "Q1", "dose": 1, "n": 500, "rate": 56.4},
    {"name": "Q2", "dose": 2, "n": 500, "rate": 59.2},
    {"name": "Q3", "dose": 3, "n": 500, "rate": 54.2},
    {"name": "Q4", "dose": 4, "n": 500, "rate": 46.2},
]


def cochran_armitage(groups: list[dict]) -> dict:
    sample_sizes = [group["n"] for group in groups]
    doses = [group["dose"] for group in groups]
    counts = [
        round(group["n"] * group["rate"] / 100.0) for group in groups
    ]
    total = sum(sample_sizes)
    responders = sum(counts)
    pooled_rate = responders / total
    score = sum(
        dose * (count - size * pooled_rate)
        for dose, count, size in zip(doses, counts, sample_sizes)
    )
    weighted_square = sum(
        size * dose * dose for size, dose in zip(sample_sizes, doses)
    )
    weighted = sum(size * dose for size, dose in zip(sample_sizes, doses))
    variance = pooled_rate * (1 - pooled_rate) * (
        weighted_square - weighted * weighted / total
    )
    z_score = score / math.sqrt(variance)
    return {
        "counts": counts,
        "n_per_bin": sample_sizes,
        "R": responders,
        "N": total,
        "pbar": pooled_rate,
        "U": score,
        "var_U": variance,
        "z": z_score,
        "p_two_sided": math.erfc(abs(z_score) / math.sqrt(2)),
    }


def main() -> None:
    result = cochran_armitage(QUARTILES)
    payload = {
        "quartiles": QUARTILES,
        "result": result,
        "note": (
            "Tests a linear trend and does not assert monotonic adjacent "
            "empirical rates."
        ),
    }
    output = ROOT / "results" / "analysis" / "grounding_trend.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
