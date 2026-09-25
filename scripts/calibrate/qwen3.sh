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
CALIB_LAYER="${CALIB_LAYER:-0}"
CALIB_SEED="${CALIB_SEED:-0}"
PERTURBATION_SEED="${PERTURBATION_SEED:-0}"
if [[ "${BUILD_CALIB_JSONL}" == "0" && ! -f "${CALIB_JSONL}" ]]; then
  echo "Custom CALIB_JSONL does not exist: ${CALIB_JSONL}" >&2
  exit 1
fi
if [[ "${BUILD_CALIB_JSONL}" == "1" ]]; then
  python -m roam.make_calibration_jsonl \
    --instances "${COCO_DIR}/annotations/instances_val2014.json" \
    --out "${CALIB_JSONL}" --n "${N_SAMPLES}" --seed "${CALIB_SEED}"
fi

EIC_OUT="${EIC_OUT:-${SCORES_ROOT}/qwen3_eic.pt}"

python -m roam.calibrate \
  --model_name "${MODEL_QWEN3}" \
  --model_type qwen3 \
  --question_file "${CALIB_JSONL}" \
  --image_folder "${COCO_DIR}/val2014" \
  --n_samples "${N_SAMPLES}" \
  --layer "${CALIB_LAYER}" \
  --seed0 "${PERTURBATION_SEED}" \
  --variance_mode env_per_example \
  --amp_dtype bf16 \
  --out "${EIC_OUT}"

echo "[done] Qwen3 EIC -> ${EIC_OUT}"
