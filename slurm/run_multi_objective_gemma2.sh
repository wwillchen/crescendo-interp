#!/bin/bash
#SBATCH --job-name=multi-gemma2
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=dcc-h200-gpu-05
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/crescendo/multi_gemma2_%j.out
#SBATCH --error=slurm/logs/crescendo/multi_gemma2_%j.err

# Gemma 2 27B: ~54GB bf16 → 1x H200 (141GB)
# Standard CausalLM — model.generate() with KV cache works natively

set -euo pipefail

MODE="${1:-crescendo}"  # "crescendo" or "direct"

PROJ_ROOT="/hpc/group/chenglab/wc187/explorations/crescendo-interp"
cd "$PROJ_ROOT"

source /hpc/group/chenglab/wc187/miniconda3/etc/profile.d/conda.sh
conda activate assistant-axis
set -a; source .env; set +a

mkdir -p slurm/logs/crescendo

export PYTHONPATH="$PROJ_ROOT/src:$PROJ_ROOT/assistant-axis:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Mode: $MODE | Model: google/gemma-2-27b-it ==="
echo "=== Started: $(date) ==="

# Background GPU monitor
gpu_monitor() {
    while true; do
        echo "[GPU] $(date +%H:%M:%S) | $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "Util: %s%% | Memory: %s/%s MiB", $1, $2, $3}')"
        sleep 30
    done
}
gpu_monitor &
GPU_MON_PID=$!
trap "kill $GPU_MON_PID 2>/dev/null" EXIT

DIRECT_FLAG=""
if [ "$MODE" = "direct" ]; then
    DIRECT_FLAG="--direct"
fi

python -m crescendo \
    --model "google/gemma-2-27b-it" \
    --objectives-file objectives/harmful_9.json \
    --vectors-dir vectors/ \
    --output-dir experiments/crescendo_runs \
    --max-turns 10 \
    --max-backtracks 0 \
    --workers 2 \
    $DIRECT_FLAG

echo "=== Finished: $(date) ==="
echo "=== Done ($MODE) ==="
