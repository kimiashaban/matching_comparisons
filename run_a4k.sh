#!/bin/bash
# Run a single MODEL across all resolutions. Uses manifest-based resume.
#
# Required: MODEL (e.g. ScaleDiff, DyPE, FreCaS)
#           or set MODEL=all to run every implemented model.
# Optional: JOB_INDEX, TOTAL_JOBS for parallel splitting
# Optional: RESOLUTIONS (space-separated, default all)
# Optional: MULTI_GPU=1 to enable multi-GPU distribution
#
# Usage:
#   MODEL=ScaleDiff sbatch run_a4k.sh
#   MODEL=all sbatch run_a4k.sh
#   MODEL=DyPE RESOLUTIONS="4096x4096" sbatch run_a4k.sh
#   MODEL=DyPE MULTI_GPU=1 sbatch run_a4k.sh
#   MODEL=DyPE JOB_INDEX=0 TOTAL_JOBS=3 sbatch run_a4k.sh
#
# Or use: ./submit_a4k.sh ScaleDiff all 3

#SBATCH --account=aip-btaati
#SBATCH --job-name=a4k
#SBATCH --gres=gpu:l40s:1  # default; submit_a4k.sh overrides via --gres
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/a4k-%j.out
#SBATCH --error=logs/a4k-%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=kimia.shaban@mail.utoronto.ca

set -e
nvidia-smi

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODEL="${MODEL:-ScaleDiff}"

if [[ -z "$MODEL" ]]; then
  echo "Error: MODEL is required. Set MODEL in this script or: MODEL=ScaleDiff sbatch run_a4k.sh" >&2
  exit 1
fi

module load opencv

cd "$WORK_DIR"
mkdir -p logs manifests

JOB_INDEX="${JOB_INDEX:-0}"
TOTAL_JOBS="${TOTAL_JOBS:-1}"
SEED="${SEED:-42}"

ALL_MODELS=(ScaleDiff I-Max HiFlow DiffuseHigh DyPE DyPE-Qwen FreCaS FreeScale)
ALL_RESOLUTIONS=(2048x4096 3072x3072 4096x4096 4096x2048)

RES_FILTER=(${RESOLUTIONS:-})

# Determine which models to run
if [[ "$MODEL" == "all" ]]; then
  RUN_MODELS=("${ALL_MODELS[@]}")
else
  RUN_MODELS=("$MODEL")
fi

get_env_for_model() {
  case "$1" in
    ScaleDiff)  echo "scalediff" ;;
    FreeScale)  echo "freescale" ;;
    DyPE-Qwen)  echo "dype_qwen" ;;
    *)          echo "diffuser_clean" ;;
  esac
}

for model in "${RUN_MODELS[@]}"; do
  env_name=$(get_env_for_model "$model")
  env_path="$HOME/envs/${env_name}/bin/activate"
  if [[ ! -f "$env_path" ]]; then
    echo "Missing environment for $model: $env_path" >&2
    echo "Create it from requirements/${env_name}.txt or adjust get_env_for_model in run_a4k.sh." >&2
    exit 1
  fi
  source "$env_path"

  for res in "${ALL_RESOLUTIONS[@]}"; do
    if (( ${#RES_FILTER[@]} > 0 )); then
      skip=1
      for r in "${RES_FILTER[@]}"; do
        [[ "$r" == "$res" ]] && { skip=0; break; }
      done
      (( skip )) && continue
    fi

    manifest="manifests/${model}_${res}.txt"
    if [[ ! -f "$manifest" ]]; then
      echo "[skip] No manifest: $manifest (run: python generate_manifests.py)"
      continue
    fi

    echo ""
    echo "========== $model $res (job $JOB_INDEX/$TOTAL_JOBS) =========="
    MULTI_GPU_FLAG=""
    [[ "${MULTI_GPU:-0}" == "1" ]] && MULTI_GPU_FLAG="--multi-gpu"

    PYTHONUNBUFFERED=1 python run_task.py \
      --model "$model" \
      --resolution "$res" \
      --seed "$SEED" \
      --job-index "$JOB_INDEX" \
      --total-jobs "$TOTAL_JOBS" \
      $MULTI_GPU_FLAG
  done
done

echo ""
echo "========== Done $MODEL (job $JOB_INDEX/$TOTAL_JOBS) =========="
