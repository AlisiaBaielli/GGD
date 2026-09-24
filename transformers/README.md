# Transformers runtime overlays

This directory contains only the model files modified by ROAM for LLaVA,
Qwen3-VL, and InternVL. It is not a complete Transformers distribution and
must not be installed as a standalone package.

Install the supported upstream Transformers version through the repository
setup script:

```bash
bash scripts/setup_env.sh
```

At runtime, `roam/transformers_fork.py` loads the required local model modules
under `transformers.models.*`. The benchmark entry points invoke this loader
before importing the affected model classes. The remaining Transformers
components come from the upstream package installed through
`requirements.txt`.

The overlaid source files retain their upstream copyright and license
headers.
