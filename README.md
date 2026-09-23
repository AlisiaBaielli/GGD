# Grounding-Guided Decoding

Grounding-Guided Decoding (GGD) is a training-free method that calibrates an environment-invariant grounding sensor offline and adaptively sharpens decoding when visual grounding weakens.

![Grounding-Guided Decoding overview](assets/method.png)

## Results

GGD reduces object hallucination across LLaVA-v1.5-7B, Qwen3-VL-8B-Instruct, and InternVL3.5-8B while preserving object recall and general multimodal capability.

![Main benchmark results](assets/main_results.png)

<details>
<summary>Qualitative examples</summary>

![Qualitative examples](assets/qualitative.png)

</details>

## Installation

```bash
bash scripts/setup_env.sh
conda activate ggd
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

## Calibration

Calibrate EIC scores once for each model:

```bash
bash scripts/calibrate/llava.sh
bash scripts/calibrate/qwen3.sh
bash scripts/calibrate/internvl.sh
```

Checkpoints are written to `scores/`.

## Evaluation

Run a single model and benchmark:

```bash
bash scripts/reproduce/run.sh <model> <benchmark>
```

Models are `llava`, `qwen3`, and `internvl`. Benchmarks are `chair`, `pope`, `amber`, and `mme`, with `mmvp` and `mmbench` also available for LLaVA and Qwen3-VL.

```bash
bash scripts/reproduce/run.sh llava chair
```

## Reproduce the main results

After calibration, reproduce the main hallucination results with:

```bash
export METHODS="vanilla vcd m3id only ggd"

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
