# Aesthetic-4K Evaluation Benchmark

Comprehensive evaluation pipeline for high-resolution image generation models (up to 4K resolution).

## Metrics

### Reference-Free (generated images + optional text prompts)

| Metric | Requires Prompts | What it measures |
|---|---|---|
| `clip_score` | Yes | CLIP cosine similarity (text-image alignment) |
| `image_reward` | Yes | Human preference reward (ImageReward-v1.0) |
| `aesthetic` | No | LAION Aesthetic Score v2 (1-10 scale) |
| `hps_v2` | Yes | Human Preference Score v2 |

### Reference-Based (generated images + real reference set)

| Metric | Requires Prompts | What it measures |
|---|---|---|
| `fid` | No | Fréchet Inception Distance (resized to 299x299) |
| `patch_fid` | No | FID on random native-resolution patches |
| `kid` | No | Kernel Inception Distance (unbiased estimator) |

## Installation

```bash
pip install -r requirements.txt
```

## Directory Layout

The script auto-discovers images from the standard output layout:

```
~/scratch/high-resolution/samples/
├── 2048x4096/
│   ├── Base-Flux/     # {index:05d}_{slug}.png
│   ├── SEGA-Flux/
│   └── ...
├── 3072x3072/
│   └── ...
└── 4096x4096/
    └── ...
```

Prompts are loaded from `sega/eval/prompts/metadata.jsonl` (JSONL, each line: `{"text": "..."}`)
and matched to images by the 5-digit numeric prefix in filenames.

## Usage

### Single model, reference-free metrics
```bash
python evaluate.py --resolution 4096x4096 --model SEGA-Flux \
    --metrics clip_score aesthetic image_reward hps_v2
```

### All reference-free metrics (shortcut)
```bash
python evaluate.py --resolution 4096x4096 --model SEGA-Flux --metrics all_free
```

### Compare multiple models
```bash
python evaluate.py --resolution 4096x4096 \
    --model SEGA-Flux SEGA-Qwen Base-Flux Base-Qwen \
    --metrics clip_score aesthetic --output results.csv
```

### Auto-discover all models at a resolution
```bash
python evaluate.py --resolution 4096x4096 --model all \
    --metrics all_free --output results.json
```

### Reference-based metrics (FID, Patch-FID, KID)
```bash
python evaluate.py --resolution 4096x4096 --model SEGA-Flux \
    --reference-dir /path/to/real_images \
    --metrics fid patch_fid kid
```

### Explicit image directory
```bash
python evaluate.py --image-dir /path/to/my/images \
    --metrics clip_score aesthetic
```

### Debugging with limited images
```bash
python evaluate.py --resolution 4096x4096 --model SEGA-Flux \
    --metrics clip_score --max-images 10 --batch-size 2
```

### Save per-image scores
```bash
python evaluate.py --resolution 4096x4096 --model SEGA-Flux \
    --metrics clip_score aesthetic \
    --save-per-image ./per_image_scores/
```

## Memory Management

4K images (4096x4096x3 = ~48MB each) require careful handling:

- Images are lazily loaded on demand
- Metrics process images in configurable batches (`--batch-size`)
- GPU memory is freed between metric computations
- Use `--batch-size 1` or `--batch-size 2` for limited GPU memory
- The `--max-images` flag allows quick tests on a subset

## Adding New Metrics

1. Create a new file in `metrics/` inheriting from `ReferenceFreeMetric` or `ReferenceBasedMetric`
2. Implement `load_model()` and `compute()`
3. Register it in `metrics/__init__.py` by adding to `METRIC_REGISTRY`

```python
from metrics.base import ReferenceFreeMetric, MetricResult

class MyNewMetric(ReferenceFreeMetric):
    name = "my_metric"
    requires_prompts = False

    def load_model(self):
        if self._model is not None:
            return
        # Load your model here
        self._model = ...

    def compute(self, images, prompts=None, batch_size=8):
        self.load_model()
        # Compute scores
        return MetricResult(name=self.name, value=mean_score, per_image=scores)
```

## Output Formats

- **Console**: Clean table with all models and metrics
- **CSV** (`--output results.csv`): Spreadsheet-friendly
- **JSON** (`--output results.json`): Programmatic access with per-metric metadata
- **Per-image JSON** (`--save-per-image dir/`): Individual scores for analysis
