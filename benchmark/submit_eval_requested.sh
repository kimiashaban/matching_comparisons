#!/bin/bash
# Evaluate the requested model set on one resolution.
#
# Usage:
#   sbatch benchmark/submit_eval_requested.sh
#   RESOLUTION=4096x4096 METRICS="all_free" sbatch benchmark/submit_eval_requested.sh

#SBATCH --account=aip-btaati
#SBATCH --job-name=a4k_eval
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RESOLUTION="${RESOLUTION:-4096x4096}"
MODELS="${MODELS:-ScaleDiff I-Max HiFlow DiffuseHigh DyPE DyPE-Qwen FreCaS FreeScale}"
METRICS="${METRICS:-clip_score aesthetic image_reward hps_v2 hps_v2_1 pick_score arniqa clipiqa liqe musiq niqe nrqm topiq_nr}"

cd "$REPO_ROOT"
mkdir -p logs results

python benchmark/evaluate.py \
  --resolution "$RESOLUTION" \
  --model $MODELS \
  --samples-root "$REPO_ROOT/outputs_by_res" \
  --prompts-path "$REPO_ROOT/data/aesthetic4k_metadata.jsonl" \
  --metrics $METRICS \
  --output "results/${RESOLUTION}_requested_metrics.csv" \
  --save-per-image "results/per_image_${RESOLUTION}"
