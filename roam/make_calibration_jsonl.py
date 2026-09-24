"""Build a reproducible, evaluation-disjoint calibration set from COCO."""
import argparse
import json
import random
import re
from pathlib import Path

PROMPTS = [
    "Describe this image.",
    "What is happening in this picture?",
    "What do you see in this image?",
    "Can you describe what's in this photo?",
]


def _image_id(value) -> int:
    if isinstance(value, int):
        return value
    match = re.search(r"(\d+)(?:\.[^.]+)?$", str(value))
    if match is None:
        raise ValueError(f"Cannot parse image ID from {value!r}")
    return int(match.group(1))


def _ids_from_record(record) -> set[int]:
    if isinstance(record, (str, int)):
        return {_image_id(record)}
    if isinstance(record, list):
        ids = set()
        for value in record:
            ids.update(_ids_from_record(value))
        return ids
    if not isinstance(record, dict):
        return set()

    ids = set()
    for key in ("image_id", "image", "file_name"):
        if key in record:
            ids.add(_image_id(record[key]))
    if "excluded_image_ids" in record:
        ids.update(_ids_from_record(record["excluded_image_ids"]))
    if "images" in record:
        ids.update(_ids_from_record(record["images"]))
    return ids


def _load_excluded_ids(path: str) -> set[int]:
    """Read IDs from JSON, JSONL, or one-ID/filename-per-line text."""
    text = Path(path).read_text().strip()
    if not text:
        return set()
    try:
        return _ids_from_record(json.loads(text))
    except json.JSONDecodeError:
        ids = set()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.update(_ids_from_record(json.loads(line)))
            except json.JSONDecodeError:
                ids.add(_image_id(line))
        return ids


def _chair_protocol_ids(images: list[dict], n: int, seed: int) -> set[int]:
    """Return the union of CHAIR IDs used by the release's model runners."""
    if n <= 0:
        return set()
    if n > len(images):
        raise ValueError(f"CHAIR exclusion requests {n} of {len(images)} images")

    # LLaVA preserves annotation order; Qwen3/InternVL sort filenames first.
    orderings = [
        [im["file_name"] for im in images],
        sorted(im["file_name"] for im in images),
    ]
    excluded = set()
    for filenames in orderings:
        random.Random(seed).shuffle(filenames)
        excluded.update(_image_id(name) for name in filenames[:n])
    return excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True,
                    help="COCO instances_*.json (provides image ids + file names).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="Limit to N images (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        help=(
            "JSON, JSONL, or text file containing evaluation image IDs or "
            "filenames. Repeat this option for multiple benchmarks."
        ),
    )
    ap.add_argument(
        "--exclude-chair-n",
        type=int,
        default=0,
        help="Exclude the union of the first N CHAIR images for all model runners.",
    )
    ap.add_argument("--exclude-chair-seed", type=int, default=3407)
    args = ap.parse_args()

    with open(args.instances) as handle:
        images = json.load(handle)["images"]

    excluded = _chair_protocol_ids(
        images, args.exclude_chair_n, args.exclude_chair_seed
    )
    for path in args.exclude_file:
        excluded.update(_load_excluded_ids(path))

    candidates = [im for im in images if int(im["id"]) not in excluded]
    random.Random(args.seed).shuffle(candidates)
    if args.n > 0:
        if len(candidates) < args.n:
            raise ValueError(
                f"Requested {args.n} calibration images, but only "
                f"{len(candidates)} remain after excluding {len(excluded)} IDs"
            )
        candidates = candidates[:args.n]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for i, im in enumerate(candidates):
            f.write(json.dumps({
                "question_id": im["id"],
                "image": im["file_name"],
                "text": PROMPTS[i % len(PROMPTS)],
            }) + "\n")
    print(
        f"wrote {len(candidates)} calibration entries -> {args.out} "
        f"(excluded {len(excluded)} evaluation IDs)"
    )

if __name__ == "__main__":
    main()
