"""Read-Only Attention Monitoring for vision-language models.

"""
from roam.monitor import (
    ROAMMonitor,
    ROAMMonitorQwen3,
    ROAMMonitorInternVL,
    ROAMLogitsProcessor,
)
from roam.grounding import compute_grounding_score_batched
from roam.sharpening import sharpen_logits
from roam.scores import tver_from_attn, compute_C, choose_intervention_layer


def apply_zscore_filter(*args, **kwargs):
    """Apply the EIC filter without eagerly importing the CLI module."""
    from roam.apply_zscore_filter import apply_zscore_filter as _apply

    return _apply(*args, **kwargs)

__all__ = [
    "ROAMMonitor",
    "ROAMMonitorQwen3",
    "ROAMMonitorInternVL",
    "ROAMLogitsProcessor",
    "compute_grounding_score_batched",
    "sharpen_logits",
    "tver_from_attn",
    "compute_C",
    "choose_intervention_layer",
    "apply_zscore_filter",
]
