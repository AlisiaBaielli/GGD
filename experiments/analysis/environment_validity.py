from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

from causal_core.envs import ENV_LIST_K7, BaseExample, EnvMaker
from causal_core.models import llava_adapter
from causal_core.scores import tver_from_attn


IMAGE_ENVIRONMENTS = ("img_mismatch", "mask", "appearance")
TEXT_ENVIRONMENTS = ("paraphrase", "neg_conflict", "ctx_rephrase")


def normalized_entropy_per_head(
    attention: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    batch, heads, _ = attention.shape
    result = torch.zeros(batch, heads, device=attention.device)
    for batch_index in range(batch):
        selected = attention[batch_index][:, mask[batch_index]]
        count = selected.shape[-1]
        if count < 2:
            continue
        selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(epsilon)
        entropy = -(selected * (selected + epsilon).log()).sum(dim=-1)
        result[batch_index] = (entropy / math.log(count)).clamp(0.0, 1.0)
    return result


def measure(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eic = torch.load(args.eic_scores, map_location="cpu", weights_only=False)
    layer = int(eic["chosen_layer"])
    causal_weights = eic["C"].float()
    eic_indices = torch.where(causal_weights > 0)[0]
    weights = causal_weights[eic_indices]
    weights = weights / weights.sum()

    with Path(os.path.expanduser(args.question_file)).open() as handle:
        questions = [json.loads(line) for line in handle if line.strip()]
    image_folder = os.path.expanduser(args.image_folder)
    examples = []
    for index, record in enumerate(questions):
        image = record.get("image")
        text = record.get("question") or record.get("text")
        if image is not None and text is not None:
            examples.append(
                BaseExample(
                    idx=index,
                    example_id=str(record.get("question_id", index)),
                    image_path=os.path.join(image_folder, image),
                    text=text,
                )
            )
    environment_maker = EnvMaker(examples, seed0=0)

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        get_model_name_from_path(model_path),
        attn_implementation="eager",
    )
    model.config.output_attentions = True
    model.config.return_dict = True
    model.to(device).eval()
    bundle = {
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "conv_mode": args.conv_mode,
        "model": model,
    }

    metrics = ("GS", "Hvis", "Htext", "TVER", "ImgMass")
    accumulators = {
        environment: {
            **{metric: 0.0 for metric in metrics},
            **{f"d_{metric}": 0.0 for metric in metrics},
            **{f"abs_{metric}": 0.0 for metric in metrics},
            "n": 0,
        }
        for environment in ENV_LIST_K7
    }

    def measure_example(image, text) -> dict:
        inputs = llava_adapter.build_inputs(bundle, image, text, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                **inputs,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        attention = output.attentions[layer][:, :, -1, :]
        vision_mask, text_mask = llava_adapter.get_kv_masks(model, inputs)
        key_values = attention.shape[-1]
        vision_mask = vision_mask[:, :key_values]
        text_mask = text_mask[:, :key_values]
        visual_entropy = normalized_entropy_per_head(
            attention.float(), vision_mask
        ).cpu()
        textual_entropy = normalized_entropy_per_head(
            attention.float(), text_mask
        ).cpu()
        tver = tver_from_attn(
            attention.float(),
            text_mask=text_mask,
            vision_mask=vision_mask,
        ).cpu()
        attention_cpu = attention.float().cpu()[0]
        image_mass = attention_cpu[:, vision_mask[0].cpu()].sum(dim=-1)
        selected_visual_entropy = visual_entropy[0, eic_indices]
        grounding_score = float(
            (1.0 - (weights * selected_visual_entropy).sum()).clamp(0, 1)
        )
        return {
            "GS": grounding_score,
            "Hvis": float(selected_visual_entropy.mean()),
            "Htext": float(textual_entropy[0, eic_indices].mean()),
            "TVER": float(tver[0, eic_indices].mean()),
            "ImgMass": float(image_mass[eic_indices].mean()),
            "ImgMass_vec": image_mass[eic_indices].tolist(),
        }

    sample_count = min(args.n_samples, len(examples))
    global_minimum_image_mass = float("inf")
    original_head_image_mass = []
    records = []
    with torch.no_grad():
        for example in tqdm(examples[:sample_count], desc="environment-validity"):
            values = {
                environment: measure_example(
                    *environment_maker.get(example, environment)
                )
                for environment in ENV_LIST_K7
            }
            original = values["orig"]
            record = {"example_id": example.example_id, "environments": {}}
            for environment, current in values.items():
                for metric in metrics:
                    accumulators[environment][metric] += current[metric]
                    delta = current[metric] - original[metric]
                    accumulators[environment][f"d_{metric}"] += delta
                    accumulators[environment][f"abs_{metric}"] += abs(delta)
                accumulators[environment]["n"] += 1
                record["environments"][environment] = {
                    metric: current[metric] for metric in metrics
                }
                record["environments"][environment]["delta_vs_orig"] = {
                    metric: current[metric] - original[metric]
                    for metric in metrics
                }
                global_minimum_image_mass = min(
                    global_minimum_image_mass,
                    min(current["ImgMass_vec"]),
                )
            original_head_image_mass.extend(original["ImgMass_vec"])
            records.append(record)

    per_environment = {}
    for environment, accumulator in accumulators.items():
        count = accumulator["n"]
        per_environment[environment] = {"n": count}
        for metric in metrics:
            per_environment[environment][metric] = accumulator[metric] / count
            per_environment[environment][f"mean_signed_{metric}"] = (
                accumulator[f"d_{metric}"] / count
            )
            per_environment[environment][f"mean_abs_{metric}"] = (
                accumulator[f"abs_{metric}"] / count
            )
    sorted_mass = sorted(original_head_image_mass)

    def percentile(percent: int) -> float | None:
        if not sorted_mass:
            return None
        index = int(round(percent / 100.0 * (len(sorted_mass) - 1)))
        return sorted_mass[max(0, min(len(sorted_mass) - 1, index))]

    return {
        "layer": layer,
        "n_eic_heads": len(eic_indices),
        "n_images": sample_count,
        "per_env": per_environment,
        "img_mass_worstcase": {
            "global_min_over_heads_images_allenvs": global_minimum_image_mass,
            "orig_min": sorted_mass[0] if sorted_mass else None,
            "orig_p1": percentile(1),
            "orig_p5": percentile(5),
            "orig_mean": (
                sum(sorted_mass) / len(sorted_mass) if sorted_mass else None
            ),
            "n_head_image_samples": len(sorted_mass),
        },
        "records": records,
    }


def paired_summary(values: list[float], bootstrap: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        array,
        size=(bootstrap, len(array)),
        replace=True,
    ).mean(axis=1)
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "wilcoxon_p": (
            float(wilcoxon(array).pvalue) if np.any(array != 0) else 1.0
        ),
    }


def analyze(payload: dict, bootstrap: int, seed: int) -> dict:
    per_environment = payload["per_env"]
    def ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if abs(denominator) > 1e-12 else None

    largest_text_visual_change = max(
        float(per_environment[environment]["mean_abs_Hvis"])
        for environment in TEXT_ENVIRONMENTS
    )
    image_visual_ratios = {
        environment: ratio(
            float(per_environment[environment]["mean_abs_Hvis"]),
            largest_text_visual_change,
        )
        for environment in IMAGE_ENVIRONMENTS
    }
    text_attention_ratios = {
        environment: ratio(
            float(per_environment[environment]["mean_abs_Htext"]),
            float(per_environment[environment]["mean_abs_Hvis"]),
        )
        for environment in TEXT_ENVIRONMENTS
    }
    image_contrasts = {environment: [] for environment in IMAGE_ENVIRONMENTS}
    text_contrasts = {environment: [] for environment in TEXT_ENVIRONMENTS}
    for record in payload["records"]:
        environments = record["environments"]
        text_visual_reference = max(
            abs(environments[environment]["delta_vs_orig"]["Hvis"])
            for environment in TEXT_ENVIRONMENTS
        )
        for environment in IMAGE_ENVIRONMENTS:
            image_contrasts[environment].append(
                abs(environments[environment]["delta_vs_orig"]["Hvis"])
                - text_visual_reference
            )
        for environment in TEXT_ENVIRONMENTS:
            text_contrasts[environment].append(
                abs(environments[environment]["delta_vs_orig"]["Htext"])
                - abs(environments[environment]["delta_vs_orig"]["Hvis"])
            )
    paired_inference = {
        "image_environment_visual_change_minus_largest_text_environment": {
            environment: paired_summary(values, bootstrap, seed + index)
            for index, (environment, values) in enumerate(image_contrasts.items())
        },
        "text_environment_text_change_minus_visual_change": {
            environment: paired_summary(values, bootstrap, seed + 10 + index)
            for index, (environment, values) in enumerate(text_contrasts.items())
        },
    }
    comparisons = [
        comparison
        for group in paired_inference.values()
        for comparison in group.values()
    ]
    def value_range(values: dict[str, float | None]) -> list[float] | None:
        defined = [value for value in values.values() if value is not None]
        return [min(defined), max(defined)] if defined else None

    return {
        "n_images": int(payload["n_images"]),
        "layer": int(payload["layer"]),
        "n_eic_heads": int(payload["n_eic_heads"]),
        "image_environment_visual_change_vs_largest_text_environment": {
            "ratios": image_visual_ratios,
            "range": value_range(image_visual_ratios),
        },
        "text_environment_text_vs_visual_change": {
            "ratios": text_attention_ratios,
            "range": value_range(text_attention_ratios),
        },
        "paired_inference": paired_inference,
        "paired_gate_all_ci95_above_zero": all(
            comparison["ci95"][0] > 0 for comparison in comparisons
        ),
    }


def write_json(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--model_name", required=True)
    measure_parser.add_argument("--eic_scores", required=True)
    measure_parser.add_argument("--question_file", required=True)
    measure_parser.add_argument("--image_folder", required=True)
    measure_parser.add_argument("--n_samples", type=int, default=200)
    measure_parser.add_argument("--conv_mode", default="llava_v1")
    measure_parser.add_argument("--output", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input", required=True)
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.add_argument("--bootstrap", type=int, default=10_000)
    analyze_parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "measure":
        write_json(args.output, measure(args))
    else:
        payload = json.loads(Path(args.input).read_text())
        write_json(args.output, analyze(payload, args.bootstrap, args.seed))


if __name__ == "__main__":
    main()
