"""Shared helpers for benchmark evaluation scripts."""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def image_id_from_filename(filename: str) -> int:
    match = re.search(r"(\d+)(?:\.[^.]+)?$", filename)
    if match is None:
        raise ValueError(f"Could not parse an image ID from {filename}")
    return int(match.group(1))


def excluded_image_ids(path: str | None) -> set[int]:
    if path is None:
        return set()
    payload = json.loads(Path(path).read_text())
    values = (
        payload.get("excluded_image_ids")
        if isinstance(payload, dict)
        else payload
    )
    if values is None:
        raise ValueError(f"No excluded_image_ids list found in {path}")
    return {int(value) for value in values}


def select_image_files(
    image_files: list[str],
    excluded_ids: set[int],
    num_samples: int,
    image_seed: int,
) -> list[str]:
    candidates = [
        filename
        for filename in image_files
        if image_id_from_filename(filename) not in excluded_ids
    ]
    if len(candidates) < num_samples:
        raise ValueError(
            f"Requested {num_samples} images, but only {len(candidates)} remain"
        )
    random.Random(image_seed).shuffle(candidates)
    return candidates[:num_samples]


def import_vcd_baseline(model: str):
    """Import VCD/M3ID helpers for Qwen3-VL or InternVL eval scripts."""
    baselines = str(REPO_ROOT / "experiments" / "baselines")
    if baselines not in sys.path:
        sys.path.insert(0, baselines)
    if model == "qwen3":
        from vcd_m3id_qwen3 import add_diffusion_noise, contrastive_generate
    elif model == "internvl":
        from vcd_m3id_internvl import add_diffusion_noise, contrastive_generate
    else:
        raise ValueError(f"unknown model for VCD import: {model}")
    return contrastive_generate, add_diffusion_noise

def load_eic_scores(
    scores_path: str,
    layer_index: Optional[int] = None,
) -> torch.Tensor:
    """Load per-head EIC scores from a calibration checkpoint."""
    payload = torch.load(scores_path, map_location="cpu")
    if isinstance(payload, dict):
        eic_scores = payload.get("C", payload.get("scores", None))
        if eic_scores is None:
            eic_scores = next(v for v in payload.values() if torch.is_tensor(v))
        if layer_index is None and "chosen_layer" in payload:
            layer_index = int(payload["chosen_layer"])
    else:
        eic_scores = payload

    if not torch.is_tensor(eic_scores):
        eic_scores = torch.tensor(eic_scores)

    if eic_scores.dim() == 2:
        if layer_index is None:
            raise ValueError("layer_index required for multi-layer score tensors")
        eic_scores = eic_scores[layer_index]

    return eic_scores.float()

def resolve_method(args) -> Tuple[str, bool]:
    """Return (method_name, needs_eic_scores) from CLI flags."""
    if getattr(args, "use_ascd", False):
        return "ascd", False
    if getattr(args, "use_only", False):
        if getattr(args, "use_eic_heads", False):
            return "only_eic", True
        return "only", False
    if getattr(args, "use_vcd", False):
        return "vcd", False
    if getattr(args, "use_m3id", False):
        return "m3id", False
    if getattr(args, "no_hook", False):
        return "vanilla", False
    return "ggd", True

def validate_method_flags(args) -> None:
    names = ("use_only", "use_vcd", "use_m3id", "use_ascd")
    active = [name for name in names if getattr(args, name, False)]
    if len(active) > 1:
        raise ValueError("Select at most one of ONLY, VCD, M3ID, or ASCD")
    if getattr(args, "use_eic_heads", False) and not getattr(
        args, "use_only", False
    ):
        raise ValueError("--use_eic_heads requires --use_only")

def caption_output_path(out_path: str, method: str) -> str:
    """Resolve caption jsonl path; out_path may be a directory or .jsonl file."""
    p = Path(out_path)
    if p.suffix == ".jsonl":
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{method}.jsonl")
