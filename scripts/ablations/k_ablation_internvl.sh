

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

K_VALUES=(1 3 5 7)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

case $K in
  1) ENVS=(orig) ;;
  3) ENVS=(orig img_mismatch mask) ;;
  5) ENVS=(orig img_mismatch mask appearance paraphrase) ;;
  7) ENVS=(orig img_mismatch mask appearance paraphrase neg_conflict ctx_rephrase) ;;
esac

CALIB_JSONL="${COCO_DIR}/calibration.jsonl"
if [ ! -f "${CALIB_JSONL}" ]; then
  echo "Missing official disjoint calibration manifest: ${CALIB_JSONL}" >&2
  echo "Create it first with scripts/calibrate/internvl.sh." >&2
  exit 1
fi

RAW="${SCORES_ROOT}/internvl_raw_K${K}.pt"
ZSCORE="${SCORES_ROOT}/internvl_eic_K${K}.pt"
ALPHA=0.7

# Set SKIP_CALIB=1 to reuse an existing ${ZSCORE} (e.g. for cheap re-runs of the
# downstream CHAIR eval without re-paying the calibration cost).
if [[ "${SKIP_CALIB:-0}" == "1" && -f "${ZSCORE}" ]]; then
  echo "Reusing existing calibration scores: ${ZSCORE}"
else
  python -m roam.calibrate \
    --model_name "${MODEL_INTERNVL}" \
    --model_type internvl \
    --n_samples "${NSAMP:-8000}" \
    --all_layers \
    --envs "${ENVS[@]}" \
    --question_file "${CALIB_JSONL}" \
    --image_folder "${COCO_DIR}/val2014" \
    --variance_mode env_per_example \
    --out "${RAW}"

  python -m roam.apply_zscore_filter --input "${RAW}" --output "${ZSCORE}"
fi

CALIB_LAYER="$(
  python -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["chosen_layer"]))' \
    "${ZSCORE}"
)"

OUT="${OUT_ROOT}/k_ablation/internvl_K${K}"
mkdir -p "${OUT}"

python experiments/chair/internvl.py \
  --seed 3407 \
  --model_path "${MODEL_INTERNVL}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}" \
  --eic_scores_path "${ZSCORE}" \
  --layer_index "${CALIB_LAYER}" \
  --alpha "${ALPHA}" \
  --num_eval_samples "${NCHAIR:-500}" \
  --method_name "roam_K${K}"

CAP="$(roam_caption_path "${OUT}" "${ALPHA}" "roam_K${K}")"
run_chair_metrics "${CAP}" "${OUT}/chair_results.json"
echo "=== InternVL K=${K} CHAIR done ==="
