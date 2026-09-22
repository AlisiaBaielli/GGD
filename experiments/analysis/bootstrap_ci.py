from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


def load_sentences(path: str) -> dict[int, dict]:
    payload = json.loads(Path(path).read_text())
    sentences = payload["sentences"]
    indexed = {int(sentence["image_id"]): sentence for sentence in sentences}
    if len(indexed) != len(sentences):
        raise ValueError(f"Duplicate image IDs in {path}")
    return indexed


def sufficient_statistics(sentences: list[dict]) -> np.ndarray:
    rows = []
    for sentence in sentences:
        generated = sentence["mscoco_generated_words"]
        hallucinated = sentence["mscoco_hallucinated_words"]
        ground_truth = set(sentence["mscoco_gt_words"])
        recalled = ground_truth.intersection(generated)
        rows.append(
            (
                int(bool(hallucinated)),
                len(hallucinated),
                len(generated),
                len(recalled),
                len(ground_truth),
            )
        )
    return np.asarray(rows, dtype=np.float64)


def pooled_metrics(statistics: np.ndarray) -> np.ndarray:
    return np.asarray(
        (
            statistics[:, 0].mean(),
            statistics[:, 1].sum() / statistics[:, 2].sum(),
            statistics[:, 3].sum() / statistics[:, 4].sum(),
        )
    )


def paired_comparison(
    reference: dict[int, dict],
    candidate: dict[int, dict],
    bootstrap: int,
    seed: int,
) -> dict:
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise ValueError(f"Image IDs differ; missing={missing[:5]} extra={extra[:5]}")

    image_ids = sorted(reference)
    reference_statistics = sufficient_statistics(
        [reference[image_id] for image_id in image_ids]
    )
    candidate_statistics = sufficient_statistics(
        [candidate[image_id] for image_id in image_ids]
    )
    reference_metrics = pooled_metrics(reference_statistics)
    candidate_metrics = pooled_metrics(candidate_statistics)
    rng = np.random.default_rng(seed)
    metric_differences = np.empty((bootstrap, 3), dtype=np.float64)
    for draw in range(bootstrap):
        indices = rng.integers(0, len(image_ids), len(image_ids))
        metric_differences[draw] = (
            pooled_metrics(candidate_statistics[indices])
            - pooled_metrics(reference_statistics[indices])
        )
    intervals = np.quantile(metric_differences, [0.025, 0.975], axis=0)
    reference_hallucinated = reference_statistics[:, 0].astype(bool)
    candidate_hallucinated = candidate_statistics[:, 0].astype(bool)
    improved = int(np.sum(reference_hallucinated & ~candidate_hallucinated))
    worsened = int(np.sum(~reference_hallucinated & candidate_hallucinated))
    discordant = improved + worsened
    mcnemar_p = (
        float(binomtest(min(improved, worsened), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    reference_recall = np.divide(
        reference_statistics[:, 3],
        reference_statistics[:, 4],
        out=np.zeros(len(image_ids)),
        where=reference_statistics[:, 4] > 0,
    )
    candidate_recall = np.divide(
        candidate_statistics[:, 3],
        candidate_statistics[:, 4],
        out=np.zeros(len(image_ids)),
        where=candidate_statistics[:, 4] > 0,
    )
    recall_difference = candidate_recall - reference_recall
    recall_p = (
        float(wilcoxon(recall_difference).pvalue)
        if np.any(recall_difference)
        else 1.0
    )
    names = ("CHAIRs", "CHAIRi", "Recall")
    return {
        "n_images": len(image_ids),
        "reference": dict(zip(names, reference_metrics.tolist())),
        "candidate": dict(zip(names, candidate_metrics.tolist())),
        "difference": dict(
            zip(names, (candidate_metrics - reference_metrics).tolist())
        ),
        "difference_ci95": {
            name: [float(intervals[0, index]), float(intervals[1, index])]
            for index, name in enumerate(names)
        },
        "mcnemar": {
            "improved": improved,
            "worsened": worsened,
            "exact_p": mcnemar_p,
        },
        "recall_wilcoxon_p": recall_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    reference = load_sentences(args.reference)
    result = {
        "reference_path": args.reference,
        "comparisons": {
            candidate: paired_comparison(
                reference,
                load_sentences(candidate),
                args.bootstrap,
                args.seed,
            )
            for candidate in args.candidates
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
