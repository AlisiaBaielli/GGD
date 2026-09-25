# ROAM: Mitigating Hallucinations in Vision-Language Models with Calibrated Grounding Monitors

Read-Only Attention Monitoring (ROAM) is a training-free decoding method that
calibrates reusable grounding monitors offline and adaptively sharpens
decoding when visual grounding weakens.

![Read-Only Attention Monitoring overview](assets/method.png)

## Results

ROAM reduces object hallucination across LLaVA-v1.5-7B, Qwen3-VL-8B-Instruct, and InternVL3.5-8B while preserving object recall and general multimodal capability.

![Main benchmark results](assets/main_results.png)

<details>
<summary>Qualitative examples</summary>

![Qualitative examples](assets/qualitative.png)

</details>

## Installation

```bash
bash scripts/setup_env.sh
conda activate roam
source scripts/_env.sh
```

The release uses Python 3.10 and installs the dependencies in `requirements.txt`. Model and dataset locations can be overridden in `scripts/_env.sh`.

## Data and model layout

Place model weights and benchmark data in the following default locations:

```text
data/
├── models/
│   ├── llava-v1.5-7b/
│   ├── Qwen3-VL-8B-Instruct/
│   └── InternVL3_5-8B-HF/
├── coco/
│   ├── annotations/
│   ├── val2014/
│   └── calibration.jsonl
├── POPE/
│   └── coco/
├── AMBER/
│   ├── image/
│   └── AMBER/
├── MME/
│   ├── MME_Benchmark_release_version/
│   └── test_merged_final.jsonl
└── mmbench/
    └── mmbench_dev_20230712.tsv
```

Download the model weights and datasets from their official sources, then place them in the directory structure shown above.

## Offline calibration

Create the calibration set and calibrate EIC scores once for each model:

```bash
bash scripts/calibrate/llava.sh
bash scripts/calibrate/qwen3.sh
bash scripts/calibrate/internvl.sh
```

Each script deterministically regenerates
`data/coco/calibration.jsonl`. By default it contains 8,000 unique
COCO images and excludes the union of the 500-image CHAIR evaluation sets and
all three POPE splits. The scripts stop if those POPE files are unavailable,
rather than silently creating a partially disjoint set. AMBER and MME use
separate image collections. The default calibration seed is `20260923`, and
the same manifest instructions are passed unchanged through each model's
native chat template. Environment perturbations use seed `0`.

Calibration uses the seven environments and the within-example variance
estimator described in the paper. The reported monitoring layers are fixed by
default: LLaVA layer 1, Qwen3-VL layer 0, and InternVL layer 1. Checkpoints in
`scores/` record the number of examples, perturbation seed, ordered image-ID
hash, variance estimator, environments, and monitoring layer.

To use different locations or sample counts, set `CALIB_JSONL`, `N_SAMPLES`,
`CALIB_LAYER`, `CALIB_SEED`, or `PERTURBATION_SEED` before running a
calibration script. A custom `CALIB_JSONL` is not regenerated, so it must
already satisfy the required evaluation-disjoint protocol.

## Evaluation

Run a single model and benchmark:

```bash
bash scripts/reproduce/run.sh <model> <benchmark>
```

Models are `llava`, `qwen3`, and `internvl`. Benchmarks are `chair`, `pope`, `amber`, and `mme`, with `mmvp` and `mmbench` also available for LLaVA and Qwen3-VL.

## Reproduce the main results

After calibration, reproduce the main hallucination results with:

```bash
export METHODS="vanilla vcd m3id only roam"

for model in llava qwen3 internvl; do
  for benchmark in chair pope amber mme; do
    bash scripts/reproduce/run.sh "${model}" "${benchmark}"
  done
done

unset METHODS
```

The reported seeds and evaluation settings are encoded in `scripts/reproduce/run.sh`. Outputs are written under `results/reproduce/`.

## Analysis

```bash
python experiments/analysis/grounding_trend.py
python experiments/analysis/environment_validity.py --help
python experiments/analysis/calibration_stability.py --help
python experiments/analysis/bootstrap_ci.py --help
python experiments/analysis/efficiency_benchmark.py --help
```

These scripts reproduce the grounding–hallucination trend, environment validation, calibration stability, paired confidence intervals, and efficiency analyses.

## Acknowledgements

This implementation builds on [LLaVA](https://github.com/haotian-liu/LLaVA) and [Hugging Face Transformers](https://github.com/huggingface/transformers). Baseline implementations are adapted from their official releases and remain subject to their original licenses.

Original code is distributed under the [MIT License](LICENSE).
