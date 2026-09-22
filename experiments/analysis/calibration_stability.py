from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from scipy.stats import pearsonr, spearmanr

from causal_core.apply_zscore_filter import apply_zscore_filter


def write_jsonl(path: str, records: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def split_calibration(args: argparse.Namespace) -> dict:
    records = []
    with Path(args.input).open() as handle:
        for line in handle:
            records.append(json.loads(line))
            if len(records) == args.n:
                break
    if len(records) != args.n:
        raise ValueError(f"Requested {args.n} records, found {len(records)}")
    image_ids = [str(record["image"]) for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Calibration prefix contains duplicate images")

    random.Random(args.seed).shuffle(records)
    midpoint = len(records) // 2
    first, second = records[:midpoint], records[midpoint:]
    first_ids = {str(record["image"]) for record in first}
    second_ids = {str(record["image"]) for record in second}
    if first_ids & second_ids:
        raise AssertionError("Calibration halves are not image-disjoint")
    write_jsonl(args.first_output, first)
    write_jsonl(args.second_output, second)
    return {
        "input_prefix_size": len(records),
        "first_size": len(first),
        "second_size": len(second),
        "shared_images": 0,
        "seed": args.seed,
    }


def load_checkpoint(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a checkpoint dictionary: {path}")
    return payload


def selected_heads(payload: dict) -> set[int]:
    scores = torch.as_tensor(payload["C"]).flatten()
    return set(torch.where(scores > 0)[0].tolist())


def image_ids(path: str) -> set[str]:
    identifiers = set()
    with Path(path).open() as handle:
        for line in handle:
            record = json.loads(line)
            value = record.get("image", record.get("question_id"))
            if value is None:
                raise ValueError(f"Missing image identifier in {path}")
            identifiers.add(str(value))
    return identifiers


def difference_summary(first: torch.Tensor, second: torch.Tensor) -> dict:
    first_tensor = torch.as_tensor(first, dtype=torch.float64).flatten()
    second_tensor = torch.as_tensor(second, dtype=torch.float64).flatten()
    if first_tensor.shape != second_tensor.shape:
        raise ValueError(
            f"Statistic shapes differ: {first_tensor.shape} vs {second_tensor.shape}"
        )
    difference = (first_tensor - second_tensor).abs()
    return {
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
    }


def agreement_summary(first: torch.Tensor, second: torch.Tensor) -> dict:
    first_array = torch.as_tensor(first, dtype=torch.float64).flatten().numpy()
    second_array = torch.as_tensor(second, dtype=torch.float64).flatten().numpy()
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"Statistic shapes differ: {first_array.shape} vs {second_array.shape}"
        )
    pearson = pearsonr(first_array, second_array)
    spearman = spearmanr(first_array, second_array)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def analyze(args: argparse.Namespace) -> dict:
    first_raw = load_checkpoint(args.first_raw)
    second_raw = load_checkpoint(args.second_raw)
    first_filtered = load_checkpoint(args.first_filtered)
    second_filtered = load_checkpoint(args.second_filtered)
    first_heads = selected_heads(first_filtered)
    second_heads = selected_heads(second_filtered)
    result = {
        "first_n_examples": int(first_raw["n_examples"]),
        "second_n_examples": int(second_raw["n_examples"]),
        "first_chosen_layer": int(first_filtered["chosen_layer"]),
        "second_chosen_layer": int(second_filtered["chosen_layer"]),
        "first_selected_heads": sorted(first_heads),
        "second_selected_heads": sorted(second_heads),
        "selected_head_jaccard": jaccard(first_heads, second_heads),
        "tver_mean_difference": difference_summary(
            first_raw["mean"], second_raw["mean"]
        ),
        "tver_variance_difference": difference_summary(
            first_raw["var"], second_raw["var"]
        ),
        "raw_eic_score_difference": difference_summary(
            first_raw["C"], second_raw["C"]
        ),
        "filtered_eic_score_difference": difference_summary(
            first_filtered["C"], second_filtered["C"]
        ),
        "filtered_eic_score_agreement": agreement_summary(
            first_filtered["C"], second_filtered["C"]
        ),
    }
    if bool(args.first_input) != bool(args.second_input):
        raise ValueError("Provide both calibration inputs or neither")
    if args.first_input:
        first_identifiers = image_ids(args.first_input)
        second_identifiers = image_ids(args.second_input)
        result["calibration_inputs"] = {
            "first_unique_images": len(first_identifiers),
            "second_unique_images": len(second_identifiers),
            "shared_images": len(first_identifiers & second_identifiers),
        }
    if args.layer is not None:
        fixed_scores = [
            apply_zscore_filter(
                torch.as_tensor(payload["mean_by_layer"])[args.layer],
                torch.as_tensor(payload["var_by_layer"])[args.layer],
            )
            for payload in (first_raw, second_raw)
        ]
        fixed_heads = [
            set(torch.where(scores > 0)[0].tolist()) for scores in fixed_scores
        ]
        fixed_result = {
            "layer": args.layer,
            "first_selected_heads": sorted(fixed_heads[0]),
            "second_selected_heads": sorted(fixed_heads[1]),
            "selected_head_jaccard": jaccard(fixed_heads[0], fixed_heads[1]),
            "tver_mean_difference": difference_summary(
                torch.as_tensor(first_raw["mean_by_layer"])[args.layer],
                torch.as_tensor(second_raw["mean_by_layer"])[args.layer],
            ),
        }
        if args.reference_filtered:
            reference_heads = selected_heads(
                load_checkpoint(args.reference_filtered)
            )
            split_union = fixed_heads[0] | fixed_heads[1]
            fixed_result["reference_selected_heads"] = sorted(reference_heads)
            fixed_result["reference_vs_split_union_jaccard"] = jaccard(
                reference_heads,
                split_union,
            )
        result["fixed_layer_comparison"] = fixed_result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--input", required=True)
    split_parser.add_argument("--first-output", required=True)
    split_parser.add_argument("--second-output", required=True)
    split_parser.add_argument("--n", type=int, default=8_000)
    split_parser.add_argument("--seed", type=int, default=3407)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--first-raw", required=True)
    analyze_parser.add_argument("--second-raw", required=True)
    analyze_parser.add_argument("--first-filtered", required=True)
    analyze_parser.add_argument("--second-filtered", required=True)
    analyze_parser.add_argument("--first-input")
    analyze_parser.add_argument("--second-input")
    analyze_parser.add_argument("--layer", type=int)
    analyze_parser.add_argument("--reference-filtered")
    analyze_parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "split":
        result = split_calibration(args)
    else:
        result = analyze(args)
    rendered = json.dumps(result, indent=2)
    if getattr(args, "output", None):
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
