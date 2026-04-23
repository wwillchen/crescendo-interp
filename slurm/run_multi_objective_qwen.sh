#!/bin/bash
#SBATCH --job-name=multi-qwen
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=dcc-h200-gpu-05
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/crescendo/multi_qwen_%j.out
#SBATCH --error=slurm/logs/crescendo/multi_qwen_%j.err

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
echo "=== Mode: $MODE | Model: Qwen/Qwen-1_8B-Chat ==="
echo "=== Started: $(date) ==="

DIRECT_FLAG=""
if [ "$MODE" = "direct" ]; then
    DIRECT_FLAG="--direct"
fi

python -m crescendo \
    --model "Qwen/Qwen-1_8B-Chat" \
    --objectives-file objectives/harmful_9.json \
    --vectors-dir vectors/ \
    --output-dir experiments/crescendo_runs \
    --max-turns 10 \
    --max-backtracks 0 \
    --workers 3 \
    $DIRECT_FLAG

echo "=== Finished: $(date) ==="
echo "=== Done ($MODE) ==="
