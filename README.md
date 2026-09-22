# Causal Decode-Time Steering for Vision-Language Models

Anonymous implementation of environment-invariant grounding-head selection and adaptive decode-time sharpening.

## Setup

```bash
bash scripts/setup_env.sh
conda activate chall
```

Place model weights and benchmark data under `data/`. Configure custom paths in `scripts/_env.sh`.

## Calibration

```bash
bash scripts/calibrate/llava.sh
bash scripts/calibrate/qwen3.sh
bash scripts/calibrate/internvl.sh
```

Calibrated EIC checkpoints are written to `scores/`.

## Evaluation

```bash
bash scripts/reproduce/run.sh <model> <benchmark>
bash scripts/reproduce/submit_all.sh
```

Models: `llava`, `qwen3`, `internvl`.

Hallucination benchmarks: `chair`, `pope`, `amber`, `mme`.

Capability benchmarks: `mmvp`, `mmbench` for LLaVA and Qwen3-VL.

Override the default method set when needed:

```bash
METHODS="vanilla chall ascd" bash scripts/reproduce/run.sh llava chair
```

ASCD is supported for LLaVA. The paper method is named `chall` in command-line interfaces.

## Analyses

```bash
python experiments/analysis/grounding_trend.py
python experiments/analysis/environment_validity.py --help
python experiments/analysis/calibration_stability.py --help
python experiments/analysis/bootstrap_ci.py --help
python experiments/analysis/efficiency_benchmark.py --help
```

Outputs are written under `results/`. Third-party code and baseline attribution are listed in `NOTICE`.
