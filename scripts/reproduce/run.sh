#!/usr/bin/env bash
# Full-scale reproduction driver for one model and benchmark.
#
#   bash scripts/reproduce/run.sh <model> <bench>
#     model : llava | qwen3 | internvl
#     bench : chair | pope | amber | mme | mmvp | mmbench
#
# Methods (decode-time): vanilla, vcd, m3id, only, only_eic (LLaVA/Qwen3 only), roam (=Ours).
# Set SKIP_EXISTING=1 to resume after partial runs.
set -uo pipefail

MODEL="${1:?usage: run.sh <model> <bench>}"
BENCH="${2:?usage: run.sh <model> <bench>}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

# ---- per-model config ----
case "${MODEL}" in
  llava)    MPATH="${MODEL_LLAVA}";    SCORES="${SCORES_ROOT}/llava_eic.pt";    LAYER=1; HAS_EIC=1 ;;
  qwen3)    MPATH="${MODEL_QWEN3}";    SCORES="${SCORES_ROOT}/qwen3_eic.pt";    LAYER=0; HAS_EIC=1 ;;
  internvl) MPATH="${MODEL_INTERNVL}"; SCORES="${SCORES_ROOT}/internvl_eic.pt"; LAYER=1; HAS_EIC=0 ;;
  *) echo "Unknown model: ${MODEL}" >&2; exit 1 ;;
esac

ALPHA="${ALPHA:-0.7}"
SEED="${SEED:-42}"
RECORD_EFFICIENCY="${RECORD_EFFICIENCY:-0}"
if [[ "${RECORD_EFFICIENCY}" == "1" ]]; then
  CHAIR_SEED="${CHAIR_SEED:-9917}"
  NCHAIR="${NCHAIR:-53}"
  : "${CHAIR_EXCLUSIONS_FILE:?Set CHAIR_EXCLUSIONS_FILE}"
else
  CHAIR_SEED="${CHAIR_SEED:-3407}"
  NCHAIR="${NCHAIR:-500}"
fi
IMAGE_SEED="${IMAGE_SEED:-${CHAIR_SEED}}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
BASE="${OUT_ROOT}/reproduce/${MODEL}_${BENCH}"
mkdir -p "${BASE}" "${OUT_ROOT}/slurm"
RUN_FAIL=0

if [[ ! -f "${SCORES}" ]]; then
  echo "Missing calibration scores: ${SCORES} (run scripts/calibrate/${MODEL}.sh)" >&2
  exit 1
fi

# Decode-time method set (override with e.g. METHODS="vanilla roam")
METHODS_REQUESTED="${METHODS:-}"
if [[ -n "${METHODS:-}" ]]; then
  read -r -a METHODS <<< "${METHODS}"
else
  METHODS=(vanilla vcd m3id only)
  [[ "${HAS_EIC}" == "1" ]] && METHODS+=(only_eic)
  METHODS+=(roam)
  [[ "${MODEL}" == "llava" ]] && METHODS+=(ascd)
fi

for method in "${METHODS[@]}"; do
  case "${method}" in
    vanilla|vcd|m3id|only|roam) ;;
    only_eic)
      [[ "${HAS_EIC}" == "1" ]] || {
        echo "Method only_eic is not supported for ${MODEL}" >&2
        exit 2
      }
      if [[ "${MODEL}" == "llava" ]] \
        && [[ "${BENCH}" == "mmvp" || "${BENCH}" == "mmbench" ]]; then
        echo "Method only_eic is not supported for LLaVA ${BENCH}" >&2
        exit 2
      fi
      ;;
    ascd)
      [[ "${MODEL}" == "llava" ]] || {
        echo "Method ascd is only supported for llava" >&2
        exit 2
      }
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 2
      ;;
  esac
done

# Flags that select the decoding method in the benchmark scripts.
# --eic_scores_path is required by the Qwen3/InternVL scripts, so we always pass it
# (it is only *used* by roam and only_eic; --no_hook disables the monitor).
method_flags() {
  local base="--eic_scores_path ${SCORES} --layer_index ${LAYER}"
  local attention=""
  local only_args=""
  [[ "${RECORD_EFFICIENCY}" == "1" && "${MODEL}" == "llava" ]] \
    && attention="--attn_implementation eager"
  if [[ "${MODEL}" == "llava" ]]; then
    if [[ "${BENCH}" == "chair" ]]; then
      only_args="--ritual_beta 0.01 --js_gamma 0.25"
    elif [[ "${BENCH}" == "pope" || "${BENCH}" == "amber" || "${BENCH}" == "mme" ]]; then
      only_args="--ritual_beta 0.1 --js_gamma 0.2"
    fi
  fi
  case "$1" in
    vanilla)  echo "${base} --no_hook ${attention}" ;;
    vcd)      echo "${base} --no_hook --use_vcd ${attention}" ;;
    m3id)     echo "${base} --no_hook --use_m3id ${attention}" ;;
    only)     echo "${base} --no_hook --use_only ${only_args} ${attention}" ;;
    only_eic) echo "${base} --no_hook --use_only --use_eic_heads ${only_args}" ;;
    roam)    echo "${base} --alpha ${ALPHA}" ;;
    ascd)
      echo "${base} --no_hook --use_ascd --ascd_alpha 1.0 --ascd_beta 0.1"
      ;;
    *) return 1 ;;
  esac
}

skip_done() { [[ "${SKIP_EXISTING}" == "1" && -f "$1" ]]; }

run_chair() {
  local selection_args=()
  [[ -n "${CHAIR_EXCLUSIONS_FILE:-}" ]] && selection_args+=(
    --exclude_image_ids_file "${CHAIR_EXCLUSIONS_FILE}"
  )
  local efficiency_args=()
  [[ "${RECORD_EFFICIENCY}" == "1" ]] && efficiency_args+=(
    --record_efficiency --per_image_seed --image_seed "${IMAGE_SEED}"
  )
  for m in "${METHODS[@]}"; do
    local out="${BASE}/${m}"; local metrics="${out}/chair_results.json"
    skip_done "${metrics}" && { echo "skip ${m}"; continue; }
    mkdir -p "${out}"
    echo "=== ${MODEL} CHAIR ${m} ==="
    python "experiments/chair/${MODEL}.py" --seed "${CHAIR_SEED}" --model_path "${MPATH}" \
      --data_path "${COCO_DIR}/val2014" --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
      --out_path "${out}" --num_eval_samples "${NCHAIR}" --max_new_tokens 128 \
      --method_name "${m}" "${selection_args[@]}" "${efficiency_args[@]}" \
      $(method_flags "${m}") || {
        echo "FAILED ${m}"
        RUN_FAIL=1
        continue
      }
    local cap; cap="$(ls -t "${out}"/*.jsonl 2>/dev/null | head -1)"
    if [[ -z "${cap}" ]]; then
      echo "no caption for ${m}"
      RUN_FAIL=1
    elif ! run_chair_metrics "${cap}" "${metrics}"; then
      echo "FAILED scoring ${m}"
      RUN_FAIL=1
    fi
  done
}

run_pope() {
  # The reported protocol uses the COCO random split only (3,000 questions).
  # Override with
  # POPE_TYPES="random popular adversarial" for the full POPE evaluation.
  local types="${POPE_TYPES:-random}"
  for m in "${METHODS[@]}"; do
    local out="${BASE}/${m}"; mkdir -p "${out}"
    for TYPE in ${types}; do
      skip_done "${out}/done_${TYPE}" && continue
      echo "=== ${MODEL} POPE ${m}/${TYPE} ==="
      if python "experiments/pope/${MODEL}.py" --seed "${SEED}" --model_path "${MPATH}" \
        --data_path "${COCO_DIR}/val2014" --pope_path "${POPE_DIR}/coco_pope_${TYPE}.json" \
        --type "${TYPE}" --dataset_name coco --out_path "${out}" \
        $(method_flags "${m}"); then
        touch "${out}/done_${TYPE}"
      else
        echo "FAILED ${m}/${TYPE}"
        RUN_FAIL=1
      fi
    done
  done
}

run_amber() {
  if [[ ! -f "${AMBER_TOOLKIT}/inference.py" ]]; then
    echo "Missing AMBER evaluator: ${AMBER_TOOLKIT}/inference.py" >&2
    echo "Set AMBER_TOOLKIT to a complete AMBER checkout." >&2
    RUN_FAIL=1
    return 1
  fi
  for m in "${METHODS[@]}"; do
    local out="${BASE}/${m}"; local infj="${out}/amber_${m}.json"; local met="${out}/amber_metrics.txt"
    skip_done "${met}" && { echo "skip ${m}"; continue; }
    mkdir -p "${out}"
    echo "=== ${MODEL} AMBER ${m} ==="
    python "experiments/amber/${MODEL}.py" --seed "${SEED}" --model_path "${MPATH}" \
      --amber_query "${AMBER_QUERY}" --amber_image_dir "${AMBER_IMAGE_DIR}" \
      --output_file "${infj}" $(method_flags "${m}") || {
        echo "FAILED ${m}"
        RUN_FAIL=1
        continue
      }
    if ! ( cd "${AMBER_TOOLKIT}" && python inference.py \
      --inference_data "${infj}" --evaluation_type g ) | tee "${met}"; then
      echo "FAILED scoring ${m}"
      RUN_FAIL=1
    fi
  done
}

run_mme() {
  for m in "${METHODS[@]}"; do
    local out="${BASE}/${m}"; local ans="${out}/mme_${m}.jsonl"; local sc="${out}/mme_score.json"
    skip_done "${sc}" && { echo "skip ${m}"; continue; }
    mkdir -p "${out}"
    echo "=== ${MODEL} MME ${m} ==="
    python "experiments/mme/${MODEL}.py" --seed "${SEED}" --model_path "${MPATH}" \
      --image_folder "${MME_IMAGE_DIR}" --question_file "${MME_QUESTIONS}" \
      --answers_file "${ans}" --max_new_tokens 8 $(method_flags "${m}") || {
        echo "FAILED ${m}"
        RUN_FAIL=1
        continue
      }
    if ! python eval/mme_score.py --answers_file "${ans}" \
      --question_file "${MME_QUESTIONS}" --out_path "${sc}"; then
      echo "FAILED scoring ${m}"
      RUN_FAIL=1
    fi
  done
}

# ---- capability benchmarks (LLaVA + Qwen3 only) ----
cap_methods() {
  if [[ -n "${METHODS_REQUESTED}" ]]; then
    echo "${METHODS_REQUESTED}"
  elif [[ "${MODEL}" == "llava" ]]; then
    echo vanilla vcd only roam ascd
  else
    echo vanilla vcd only roam
  fi
}
# CAP_LIMIT (optional): cap #questions for cheap validation; empty = full set.
CAP_LIMIT="${CAP_LIMIT:-}"
cap_limit_flag() { [[ -n "${CAP_LIMIT}" ]] && echo "--limit ${CAP_LIMIT}"; }
# Capability scripts: LLaVA uses *_eval.py, Qwen3 uses *_qwen3.py.
cap_suffix() { [[ "${MODEL}" == "llava" ]] && echo "eval" || echo "${MODEL}"; }
CAP_OK=0; CAP_FAIL=0

run_mmvp() {
  for m in $(cap_methods); do
    local out="${BASE}/${m}"; mkdir -p "${out}"
    skip_done "${out}/${m}_summary.json" && { echo "skip ${m}"; continue; }
    local extra="--method_name ${m} $(cap_limit_flag)"
    case "${m}" in
      vanilla)  extra+=" --alpha 0" ;;
      vcd)      extra+=" --use_vcd" ;;
      only)     extra+=" --use_only" ;;
      only_eic) extra+=" --use_only --use_eic_heads" ;;
      roam)    extra+=" --alpha ${ALPHA}" ;;
      ascd)     extra+=" --use_ascd --ascd_alpha 1.0 --ascd_beta 0.1" ;;
    esac
    echo "=== ${MODEL} MMVP ${m} ==="
    if python "experiments/analysis/mmvp_$(cap_suffix).py" --model_path "${MPATH}" \
      --eic_scores_path "${SCORES}" --layer_index "${LAYER}" --out_path "${out}" ${extra}; then
      CAP_OK=$((CAP_OK+1))
    else
      echo "FAILED ${m}"
      CAP_FAIL=$((CAP_FAIL+1))
      RUN_FAIL=1
    fi
  done
}

run_mmbench() {
  local tsv="${MMBENCH_TSV:-${REPO_ROOT}/data/mmbench/mmbench_dev_20230712.tsv}"
  for m in $(cap_methods); do
    local out="${BASE}/${m}"; mkdir -p "${out}"
    skip_done "${out}/mmbench_${m}.json" && { echo "skip ${m}"; continue; }
    echo "=== ${MODEL} MMBench ${m} ==="
    if python "experiments/analysis/mmbench_$(cap_suffix).py" --model_path "${MPATH}" \
      --question_file "${tsv}" --eic_scores_path "${SCORES}" --layer_index "${LAYER}" \
      --alpha "${ALPHA}" --method "${m}" --out_dir "${out}" $(cap_limit_flag); then
      CAP_OK=$((CAP_OK+1))
    else
      echo "FAILED ${m}"
      CAP_FAIL=$((CAP_FAIL+1))
      RUN_FAIL=1
    fi
  done
}

# ---- mmvp/mmbench guard ----
if [[ "${BENCH}" == "mmvp" || "${BENCH}" == "mmbench" ]] && [[ "${MODEL}" == "internvl" ]]; then
  echo "Capability benchmarks (${BENCH}) are available for LLaVA/Qwen3 only." >&2
  exit 1
fi

case "${BENCH}" in
  chair)   run_chair ;;
  pope)    run_pope ;;
  amber)   run_amber ;;
  mme)     run_mme ;;
  mmvp)    run_mmvp ;;
  mmbench) run_mmbench ;;
  *) echo "Unknown bench: ${BENCH}" >&2; exit 1 ;;
esac

echo ""
echo "================ ${MODEL} ${BENCH} table ================"
BASE="${BASE}" BENCH="${BENCH}" python - <<'PY'
import json, os, glob
base = os.environ["BASE"]; bench = os.environ["BENCH"]
methods = ["vanilla", "vcd", "m3id", "only", "only_eic", "roam", "ascd"]
def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
if bench == "chair":
    print(f"{'method':<10}{'CHAIRs':>9}{'CHAIRi':>9}{'Recall':>9}")
    for m in methods:
        d = jload(os.path.join(base, m, "chair_results.json"))
        if not d: continue
        om = d.get("overall_metrics", d)
        print(f"{m:<10}{om.get('CHAIRs',0)*100:9.1f}{om.get('CHAIRi',0)*100:9.1f}{om.get('Recall',0)*100:9.1f}")
elif bench == "pope":
    print(f"{'method':<10}{'Acc(avg)':>10}{'F1(avg)':>10}")
    for m in methods:
        fs = glob.glob(os.path.join(base, m, "pope_*.json"))
        if not fs: continue
        accs=[]; f1s=[]
        for f in fs:
            d = jload(f)
            if d: accs.append(d.get("accuracy",0)); f1s.append(d.get("f1",0))
        if accs:
            print(f"{m:<10}{sum(accs)/len(accs):10.2f}{sum(f1s)/len(f1s):10.2f}")
elif bench == "amber":
    print("(AMBER metrics per method below — see amber_metrics.txt)")
    for m in methods:
        p = os.path.join(base, m, "amber_metrics.txt")
        if os.path.exists(p):
            print(f"-- {m} --"); print(open(p).read().strip())
elif bench == "mme":
    print(f"{'method':<10}{'MME(/800)':>11}{'Percept':>10}{'Cognition':>11}")
    for m in methods:
        d = jload(os.path.join(base, m, "mme_score.json"))
        if not d: continue
        print(f"{m:<10}{d.get('mme_total',0):11.2f}{d.get('perception_total',0):10.2f}{d.get('cognition_total',0):11.2f}")
elif bench == "mmvp":
    print(f"{'method':<10}{'Single%':>9}{'Pair%':>9}")
    for m in ["vanilla","vcd","only","roam","ascd"]:
        d = jload(os.path.join(base, m, f"{m}_summary.json"))
        if not d: continue
        print(f"{m:<10}{d.get('single_acc',0)*100:9.2f}{d.get('pair_acc',0)*100:9.2f}")
elif bench == "mmbench":
    print(f"{'method':<10}{'Acc%':>9}")
    for m in ["vanilla","vcd","only","roam","ascd"]:
        d = jload(os.path.join(base, m, f"mmbench_{m}.json"))
        if not d: continue
        print(f"{m:<10}{d.get('accuracy',0):9.2f}")
PY
echo "Outputs under ${BASE}"

# Capability benches: fail loudly if no method produced output.
if [[ "${BENCH}" == "mmvp" || "${BENCH}" == "mmbench" ]] && [[ "${CAP_OK}" -eq 0 ]]; then
  echo "ERROR: all ${BENCH} methods failed (${CAP_FAIL} failures)" >&2
  exit 1
fi

if [[ "${RUN_FAIL}" -ne 0 ]]; then
  echo "ERROR: one or more ${MODEL} ${BENCH} runs failed" >&2
  exit 1
fi
