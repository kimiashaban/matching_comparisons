#!/bin/bash
# Focused rerun for metrics that were missing from the full Qwen evaluation.
# Defaults to a one-image smoke test; set MAX_IMAGES="" for the full 195/image run.
# Example:
#   sbatch --time=00:25:00 benchmark/submit_eval_qwen_missing.sh
#   MAX_IMAGES="" METRICS="image_reward arniqa clipiqa liqe" sbatch benchmark/submit_eval_qwen_missing.sh

#SBATCH --job-name=a4k_miss
#SBATCH --account=def-btaati_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:25:00
#SBATCH --output=/home/kimias/links/scratch/matching_comparisons/logs/eval-missing-%j.out
#SBATCH --error=/home/kimias/links/scratch/matching_comparisons/logs/eval-missing-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kimia.shaban@mail.utoronto.ca
#SBATCH --exclude=rg31901

set -euo pipefail

module load StdEnv/2023 gcc/12.3 python/3.11 arrow/24.0.0 opencv/4.13.0 2>/dev/null \
  || module load python/3.11.5 cuda/12.2 arrow opencv/4.13.0 2>/dev/null \
  || true
source ~/links/scratch/envs/qwen-eval/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HUB_OFFLINE="${OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

REPO=~/links/scratch/matching_comparisons
cd "$REPO"
mkdir -p logs results

RESOLUTION="${RESOLUTION:-4096x4096}"
MODELS="${MODELS:-Dype-Qwen Ultraimage-Qwen}"
METRICS="${METRICS:-image_reward hps_v2 arniqa clipiqa liqe topiq_nr}"
OUTPUT_TAG="${OUTPUT_TAG:-qwen_missing_smoke}"
MAX_IMAGES="${MAX_IMAGES:-1}"

EXTRA_ARGS=()
if [[ -n "$MAX_IMAGES" ]]; then
  EXTRA_ARGS+=(--max-images "$MAX_IMAGES")
fi

echo "[missing] start $(date -Is) host=$(hostname)"
python - <<'PY'
import torch
print(f"[missing] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[missing] gpu={torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("[missing] ERROR: CUDA is not available")
PY

python - <<'PY'
mods = ["datasets", "pyarrow", "cv2", "ImageReward", "hpsv2", "pyiqa"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"[missing] import {name}: OK {getattr(mod, '__version__', '')}")
    except Exception as exc:
        raise SystemExit(f"[missing] import {name}: FAIL {exc!r}")
PY

python benchmark/evaluate.py \
  --resolution "$RESOLUTION" \
  --model $MODELS \
  --samples-root "$REPO/outputs_by_res" \
  --prompts-path "$REPO/data/aesthetic4k_metadata.jsonl" \
  --metrics $METRICS \
  --batch-size "${BATCH_SIZE:-1}" \
  "${EXTRA_ARGS[@]}" \
  --output "results/${RESOLUTION}_${OUTPUT_TAG}_metrics.csv" \
  --save-per-image "results/per_image_${RESOLUTION}_${OUTPUT_TAG}"

echo "[missing] done $(date -Is)"
