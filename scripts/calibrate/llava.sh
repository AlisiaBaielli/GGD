#!/bin/bash
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

if [[ -z "${CALIB_JSONL:-}" ]]; then
  CALIB_JSONL="${COCO_DIR}/calibration.jsonl"
  BUILD_CALIB_JSONL=1
else
  BUILD_CALIB_JSONL=0
fi
N_SAMPLES="${N_SAMPLES:-8000}"
CALIB_LAYER="${CALIB_LAYER:-1}"
CALIB_SEED="${CALIB_SEED:-20260923}"
PERTURBATION_SEED="${PERTURBATION_SEED:-0}"
if [[ "${BUILD_CALIB_JSONL}" == "0" && ! -f "${CALIB_JSONL}" ]]; then
  echo "Custom CALIB_JSONL does not exist: ${CALIB_JSONL}" >&2
  exit 1
fi
if [[ "${BUILD_CALIB_JSONL}" == "1" ]]; then
  EXCLUDE_ARGS=(
    --exclude-chair-n "${CHAIR_EVAL_SAMPLES:-500}"
    --exclude-chair-seed "${CHAIR_EVAL_SEED:-3407}"
  )
  for split in random popular adversarial; do
    pope_file="${POPE_DIR}/coco_pope_${split}.json"
    if [[ ! -f "${pope_file}" ]]; then
      echo "Missing POPE split required for disjoint calibration: ${pope_file}" >&2
      exit 1
    fi
    EXCLUDE_ARGS+=(--exclude-file "${pope_file}")
  done
  python -m roam.make_calibration_jsonl \
    --instances "${COCO_DIR}/annotations/instances_val2014.json" \
    --out "${CALIB_JSONL}" --n "${N_SAMPLES}" --seed "${CALIB_SEED}" \
    "${EXCLUDE_ARGS[@]}"
fi

RAW="${RAW:-${SCORES_ROOT}/llava_raw.pt}"
ZSCORE="${ZSCORE:-${SCORES_ROOT}/llava_eic.pt}"

python -m roam.calibrate \
  --model_name "${MODEL_LLAVA}" \
  --model_type llava \
  --question_file "${CALIB_JSONL}" \
  --image_folder "${COCO_DIR}/val2014" \
  --n_samples "${N_SAMPLES}" \
  --layer "${CALIB_LAYER}" \
  --seed0 "${PERTURBATION_SEED}" \
  --variance_mode env_per_example \
  --out "${RAW}"

python -m roam.apply_zscore_filter \
  --input "${RAW}" \
  --output "${ZSCORE}"

echo "[done] LLaVA EIC -> ${ZSCORE}"
