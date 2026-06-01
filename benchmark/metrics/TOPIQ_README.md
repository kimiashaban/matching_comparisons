# TOPIQ-NR: No-Reference Image Quality Assessment

TOPIQ-NR (IEEE TIP 2024) is a state-of-the-art no-reference image quality assessment metric.
It measures **technical quality** — sharpness, detail preservation, artifacts, texture
degradation — using cross-scale attention on a ResNet50 backbone.

- **Score range:** 0–1 (higher is better)
- **Reference-free:** only needs generated images, no prompts or reference set
- **Benchmark:** SRCC = 0.93 on KonIQ-10k

## Installation

### Generic (non-ComputeCanada)

```bash
pip install pyiqa
```

This pulls in all dependencies (`timm`, `facexlib`, `opencv-python`, etc.) automatically.

### ComputeCanada / Alliance Clusters

ComputeCanada blocks `opencv-python` via a dummy pip wheel.
OpenCV must be loaded as a **system module before** activating the virtualenv.

```bash
# 1. Load modules (BEFORE activating venv)
module load StdEnv/2023 gcc/12.3 opencv/4.10.0

# 2. Activate your virtualenv
source ~/envs/benchmark/bin/activate

# 3. Install pyiqa without deps (avoids the opencv dummy wheel error)
pip install pyiqa --no-deps

# 4. Install the two runtime deps that pyiqa needs
pip install 'timm>=0.9' facexlib --no-deps
```

> **Note on timm upgrade:** `pyiqa`'s TOPIQ architecture requires `timm >= 0.9`
> (uses `timm.models._builder`). Upgrading from `timm==0.6.13` to `1.0.x` is safe —
> `image-reward` and `hpsv2` continue to work (tested).

### Verify Installation

```bash
# On ComputeCanada, make sure opencv is loaded first
module load StdEnv/2023 gcc/12.3 opencv/4.10.0
source ~/envs/benchmark/bin/activate

python -c "
import pyiqa, torch
m = pyiqa.create_metric('topiq_nr', device='cuda')
score = m(torch.rand(1, 3, 512, 512).to('cuda'))
print(f'topiq_nr OK: {score.item():.4f}')
"
```

Model weights (~173 MB) are downloaded automatically on first run to `~/.cache/torch/hub/pyiqa/`.

## Integration into the Benchmark Pipeline

### Step 1: Add the metric file

Place `topiq_metric.py` in the `metrics/` directory alongside the other metric files.

### Step 2: Register in `metrics/__init__.py`

Add the import and registry entry:

```python
from .topiq_metric import TOPIQMetric          # add with other imports

METRIC_REGISTRY: dict[str, type[Metric]] = {
    ...
    "topiq_nr": TOPIQMetric,                   # add to the registry dict
    ...
}
```

No changes to `evaluate.py` are needed — the metric is auto-discovered from the registry
and automatically included in the `all` and `all_free` aliases.

### Step 3: Run

```bash
# Single metric
python evaluate.py --resolution 6144x6144 --model SEGA-Flux --metrics topiq_nr

# Combined with other metrics
python evaluate.py --resolution 6144x6144 --model SEGA-Flux SEGA-Qwen \
    --metrics topiq_nr aesthetic clip_score

# All reference-free metrics (topiq_nr included automatically)
python evaluate.py --resolution 6144x6144 --model all --metrics all_free
```

## How It Works

1. Each input image is resized to **512 px** on its longest side (LANCZOS resampling),
   matching the resolution range the model was trained on.
2. The resized image is converted to a `[0, 1]` float tensor and passed through TOPIQ-NR.
3. Per-image scores are collected and averaged across the dataset.

Output follows the standard `MetricResult` format with `value` (mean), `per_image` scores,
and `extra` metadata.

## ComputeCanada Runtime Reminder

Every time you run the benchmark with `topiq_nr`, the opencv module must be loaded
**before** activating the virtualenv:

```bash
module load StdEnv/2023 gcc/12.3 opencv/4.10.0
source ~/envs/benchmark/bin/activate
python evaluate.py ...
```

If you load modules after activating the venv, Python may resolve to the system
interpreter instead of the venv one, causing `ModuleNotFoundError: No module named 'torch'`.

## Reference

- **Paper:** Chen et al., "TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment", IEEE TIP 2024
- **Library:** [IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch)
