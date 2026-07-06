#!/bin/bash
# Evaluate Dype-Qwen + Ultraimage-Qwen at 4096x4096 using the qwen-eval env.
# Model-level aggregate metrics across all 195 images, plus per-image JSON.
#SBATCH --job-name=a4k_eval
#SBATCH --account=def-btaati_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/home/kimias/links/scratch/matching_comparisons/logs/eval-%j.out
#SBATCH --error=/home/kimias/links/scratch/matching_comparisons/logs/eval-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kimia.shaban@mail.utoronto.ca
#SBATCH --exclude=rg31901

set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" && -n "${TIME_LIMIT:-}" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" TimeLimit="$TIME_LIMIT"
fi

module load StdEnv/2023 gcc/12.3 python/3.11 arrow/24.0.0 opencv/4.13.0 2>/dev/null \
  || module load python/3.11.5 cuda/12.2 arrow opencv/4.13.0 2>/dev/null \
  || true
source ~/links/scratch/envs/qwen-eval/bin/activate

# Metric weights are cached under ~/.cache/huggingface. Compute nodes do not
# have internet access, so keep Transformers/HF in offline cache mode.
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
METRICS="${METRICS:-clip_score aesthetic image_reward hps_v2 hps_v2_1 pick_score arniqa clipiqa liqe musiq niqe nrqm topiq_nr}"
SAMPLES_ROOT="$HOME/links/scratch/high-resolution/samples"
EVAL_SAMPLES_ROOT="$REPO/outputs_by_res"
OUTPUT_TAG="${OUTPUT_TAG:-qwen}"
EXTRA_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  EXTRA_ARGS+=(--max-images "$MAX_IMAGES")
fi

echo "[eval] start $(date -Is) host=$(hostname)"
python - <<'PY'
import torch
print(f"[eval] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[eval] gpu={torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("[eval] ERROR: CUDA is not available; refusing to run slow CPU evaluation")
PY

ckpt="$TORCH_HOME/hub/checkpoints/resnet50-0676ba61.pth"
if [[ -f "$ckpt" && ! -s "$ckpt" ]]; then
  echo "[eval] removing zero-byte checkpoint: $ckpt"
  rm -f "$ckpt"
fi

for model in $MODELS; do
  img_dir="$SAMPLES_ROOT/$RESOLUTION/$model/aesthetic-4k"
  count=$(find "$img_dir" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)
  echo "[eval] $model images: $count at $img_dir"
  if [[ "$count" -eq 0 ]]; then
    echo "[eval] ERROR: no images found for $model" >&2
    exit 1
  fi
done

python benchmark/evaluate.py \
  --resolution "$RESOLUTION" \
  --model $MODELS \
  --samples-root "$EVAL_SAMPLES_ROOT" \
  --prompts-path "$REPO/data/aesthetic4k_metadata.jsonl" \
  --metrics $METRICS \
  --batch-size "${BATCH_SIZE:-1}" \
  "${EXTRA_ARGS[@]}" \
  --output "results/${RESOLUTION}_${OUTPUT_TAG}_metrics.csv" \
  --save-per-image "results/per_image_${RESOLUTION}_${OUTPUT_TAG}"
echo "[eval] done $(date -Is)"
