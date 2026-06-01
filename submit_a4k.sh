#!/bin/bash
# Submit job(s) for one model (or "all").
#
# Usage:
#   ./submit_a4k.sh MODEL [RESOLUTIONS] [N_JOBS] [N_GPUS] [HOURS] [GPU_TYPE] [SEED]

MODEL=$1
RESOLUTIONS=${2:-}
N_JOBS=${3:-1}
HOURS=${5:-12}
GPU_TYPE=${6:-l40s}
SEED=${7:-12345}

# DyPE-Qwen: prefer 4 GPUs at high res (override with 4th arg, e.g. 2)
if [[ "$MODEL" == "DyPE-Qwen" ]]; then
  N_GPUS=${4:-4}
else
  N_GPUS=${4:-1}
fi

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 MODEL [RESOLUTIONS] [N_JOBS] [N_GPUS] [HOURS] [GPU_TYPE] [SEED]"
  echo ""
  echo "  MODEL:       ScaleDiff, I-Max, HiFlow, DiffuseHigh, DyPE, DyPE-Qwen,"
  echo "               FreCaS, FreeScale, or all"
  echo "  RESOLUTIONS: space-separated, e.g. \"4096x4096\" or \"4096x4096 2048x4096\""
  echo "               use \"all\" or \"\" for all resolutions (default: all)"
  echo "  N_JOBS:      number of parallel jobs (default 1)"
  echo "  N_GPUS:      GPUs per job (default 1; >=2 enables --multi_gpu)"
  echo "  HOURS:       time limit in hours (default 12)"
  echo "  GPU_TYPE:    l40s or h100 (default l40s)"
  echo "  SEED:        generation seed (default 12345)"
  echo ""
  echo "Examples:"
  echo "  $0 ScaleDiff                                     # all res, 1 job, 1 GPU, 12h, l40s"
  echo "  $0 DyPE 4096x4096                                # only 4096x4096"
  echo "  $0 DyPE \"4096x4096 3072x3072\"                    # two resolutions"
  echo "  $0 DyPE all 3                                     # all res, 3 parallel jobs"
  echo "  $0 DyPE 4096x4096 3 2                             # 3 jobs, 2 GPUs (multi-GPU)"
  echo "  $0 HiFlow 4096x4096 1 2 8 h100 12345             # 2x H100, 8h"
  echo "  $0 all all 2 2 12 h100                            # all models, all res, 2 jobs, 2x H100"
  exit 1
fi

# Treat "all" as empty (= all resolutions)
[[ "$RESOLUTIONS" == "all" ]] && RESOLUTIONS=""

cd "$(dirname "$0")"
mkdir -p logs 2>/dev/null || true

MULTI_GPU=0
(( N_GPUS > 1 )) && MULTI_GPU=1

TIME_FMT="${HOURS}:00:00"

for i in $(seq 0 $((N_JOBS - 1))); do
  job_name="a4k_${MODEL}_${i}"
  echo "Submitting $job_name (job $i of $N_JOBS for $MODEL, ${N_GPUS}x ${GPU_TYPE}, ${HOURS}h, seed ${SEED})"
  sbatch -J "$job_name" \
    --gres=gpu:${GPU_TYPE}:${N_GPUS} \
    --time=${TIME_FMT} \
    --export=ALL,MODEL="$MODEL",JOB_INDEX=$i,TOTAL_JOBS=$N_JOBS,MULTI_GPU=$MULTI_GPU,RESOLUTIONS="$RESOLUTIONS",SEED="$SEED" \
    run_a4k.sh
done
echo "Submitted $N_JOBS job(s) for $MODEL (${N_GPUS}x ${GPU_TYPE}, ${HOURS}h, seed ${SEED}). Check: squeue -u \$USER"
