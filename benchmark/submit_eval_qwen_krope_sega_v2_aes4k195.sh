#!/bin/bash
# Evaluate the combined v2 Qwen K-RoPE+SEGA Aesthetic-4K folder.

#SBATCH --job-name=eval_qks_v2
#SBATCH --account=def-btaati
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=logs/eval_qks_v2_%j.out
#SBATCH --error=logs/eval_qks_v2_%j.err

set -euo pipefail

module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/24.0.0 opencv/4.13.0 2>/dev/null \
  || module load python/3.11.5 cuda/12.2 arrow opencv/4.13.0 2>/dev/null \
  || true
source /home/kimias/links/scratch/envs/qwen-eval/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

REPO="/lustre10/scratch/kimias/matching_comparisons"
cd "$REPO"
mkdir -p logs results

MODEL="Qwen-K-Rope-SEGA-qwenheuristic-v2-aes4k195"
RESOLUTION="4096x4096"
SAMPLES_ROOT="$REPO/outputs_by_res"
IMG_DIR="$SAMPLES_ROOT/$RESOLUTION/$MODEL"
METRICS="clip_score aesthetic image_reward hps_v2 hps_v2_1 pick_score arniqa clipiqa liqe musiq niqe nrqm topiq_nr"

echo "[eval_qks_v2] start $(date -Is) host=$(hostname)"
echo "[eval_qks_v2] model=$MODEL"
echo "[eval_qks_v2] img_dir=$IMG_DIR"

count=$(find "$IMG_DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)
echo "[eval_qks_v2] images=$count"
if [[ "$count" -ne 195 ]]; then
  echo "[eval_qks_v2] ERROR: expected 195 images, found $count" >&2
  exit 1
fi

python - <<'PY'
import torch
print(f"[eval_qks_v2] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[eval_qks_v2] gpu={torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("[eval_qks_v2] ERROR: CUDA is not available")
PY

ckpt="$TORCH_HOME/hub/checkpoints/resnet50-0676ba61.pth"
if [[ -f "$ckpt" && ! -s "$ckpt" ]]; then
  echo "[eval_qks_v2] removing zero-byte checkpoint: $ckpt"
  rm -f "$ckpt"
fi

python benchmark/evaluate.py \
  --resolution "$RESOLUTION" \
  --model "$MODEL" \
  --samples-root "$SAMPLES_ROOT" \
  --prompts-path "$REPO/data/aesthetic4k_metadata.jsonl" \
  --metrics $METRICS \
  --batch-size 1 \
  --output "results/${RESOLUTION}_${MODEL}_metrics.csv" \
  --save-per-image "results/per_image_${RESOLUTION}_${MODEL}"

echo "[eval_qks_v2] done $(date -Is)"
