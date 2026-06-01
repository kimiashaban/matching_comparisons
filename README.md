# matching_comparisons

Self-contained repo for generating Aesthetic-4K samples and evaluating:

`ScaleDiff`, `I-Max`, `HiFlow`, `DiffuseHigh`, `DyPE`, `DyPE-Qwen`, `FreCaS`, `FreeScale`.

This repo includes the runner, metrics code, the Aesthetic-4K prompt metadata, the zero-shot prompt metadata, and lightweight copies of the external model repos. Generated outputs and model checkpoints are intentionally excluded.

## Layout

```text
data/
  aesthetic4k_metadata.jsonl
  zero_shot_high_res_image_gen_benchmark.jsonl
external_repos/
  ScaleDiff/ I-Max/ HiFlow/ DiffuseHigh/ DyPE/ FreCaS/ FreeScale/
benchmark/
  evaluate.py
  metrics/
requirements/
  model-specific requirement snapshots
```

## Setup

Use Python 3.10 or 3.11. The model repos do not all agree on package versions, so separate environments are safest. For comparison, I used one for everything, with the exception of a separate env for DyPE-Qwen and a separate one for FreeScale/FreCas (as they use a different base method for generation). 

```bash
cd matching_comparisons

python -m venv ~/envs/diffuser_clean
source ~/envs/diffuser_clean/bin/activate
pip install --upgrade pip
pip install -r requirements/dype.txt
pip install -r requirements/hiflow.txt
pip install -r requirements/diffusehigh.txt
pip install -r requirements_all_models.txt

python -m venv ~/envs/freescale
source ~/envs/freescale/bin/activate
pip install --upgrade pip
pip install -r requirements/freescale.txt

python -m venv ~/envs/scalediff
source ~/envs/scalediff/bin/activate
pip install --upgrade pip
pip install -r requirements/scalediff.txt

python -m venv ~/envs/dype_qwen
source ~/envs/dype_qwen/bin/activate
pip install --upgrade pip
pip install -r requirements/dype.txt

python -m venv ~/envs/metrics
source ~/envs/metrics/bin/activate
pip install --upgrade pip
pip install -r requirements/metrics.txt
```

Before running, make sure Hugging Face access is set up for gated models used by the repos, especially FLUX and SDXL:

```bash
huggingface-cli login
```

## Generate Manifests

```bash
cd matching_comparisons
source ~/envs/diffuser_clean/bin/activate
python generate_manifests.py
```

This creates `manifests/<MODEL>_<RESOLUTION>.txt`. The Aesthetic-4K metadata file has 195 prompts.

## Generate Images

Dry-run one task first:

```bash
python run_task.py --model ScaleDiff --resolution 4096x4096 --seed 12345 --dry-run
```

Run one model locally or inside an interactive GPU allocation:

```bash
MODEL=ScaleDiff RESOLUTIONS="4096x4096" SEED=12345 bash run_a4k.sh
```

Submit to SLURM:

```bash
./submit_a4k.sh ScaleDiff "4096x4096" 1 1 12 l40s 12345
./submit_a4k.sh DyPE-Qwen "4096x4096" 1 4 12 h100 12345
```

Run all requested models:

```bash
./submit_a4k.sh all "4096x4096" 1 1 12 l40s
```

Outputs are written to:

```text
outputs_by_res/<resolution>/<model>/
```

The runner resumes automatically. If an output file beginning with `00042_` already exists, prompt index 42 is skipped.

The default generation seed is `12345`, chosen so these images do not duplicate the previous project's seed-0 outputs. Override with `--seed` for `run_task.py` or `SEED=<value>` for `run_a4k.sh` / `submit_a4k.sh`.

## Evaluate

Use the metrics environment:

```bash
source ~/envs/metrics/bin/activate
python benchmark/evaluate.py \
  --resolution 4096x4096 \
  --model ScaleDiff I-Max HiFlow DiffuseHigh DyPE DyPE-Qwen FreCaS FreeScale \
  --samples-root outputs_by_res \
  --prompts-path data/aesthetic4k_metadata.jsonl \
  --metrics clip_score aesthetic image_reward hps_v2 hps_v2_1 pick_score arniqa clipiqa liqe musiq niqe nrqm topiq_nr \
  --output results/4096x4096_requested_metrics.csv \
  --save-per-image results/per_image_4096x4096
```

Or submit the helper:

```bash
RESOLUTION=4096x4096 sbatch benchmark/submit_eval_requested.sh
```

Reference-based metrics (`fid`, `patch_fid`, `kid`) need real Aesthetic-4K images, which are not included here. Run them only if you have a reference image directory:

```bash
python benchmark/evaluate.py \
  --resolution 4096x4096 \
  --model ScaleDiff \
  --reference-dir /path/to/Aesthetic-4K/eval/size_4096/images \
  --metrics fid patch_fid kid
```

## Run Checklist

1. Clone this repo and create the environments above.
2. Log in to Hugging Face and confirm gated model access.
3. Run `python generate_manifests.py`.
4. Dry-run one model with `run_task.py`.
5. Generate all requested model outputs for the target resolution(s).
6. Run `benchmark/evaluate.py` on the generated folders.
7. Put final CSVs and per-image JSON files under `results/`.
