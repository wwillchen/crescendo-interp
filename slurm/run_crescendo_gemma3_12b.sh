#!/bin/bash
#SBATCH --job-name=cresc-gemma3-12b
#SBATCH --partition=gpu-common
#SBATCH --account=chenglab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/crescendo/gemma3_12b_%j.out
#SBATCH --error=slurm/logs/crescendo/gemma3_12b_%j.err

# Gemma 3 12B: ~24GB bf16 → 1x RTX 5000 Ada (32GB) on gpu-common
# Standard CausalLM — model.generate() with KV cache works natively
# No pre-computed vectors yet — activations saved as raw .pt for later projection

set -euo pipefail

PROJ_ROOT="/hpc/group/chenglab/wc187/explorations/crescendo-interp"
cd "$PROJ_ROOT"

source /hpc/group/chenglab/wc187/miniconda3/etc/profile.d/conda.sh
conda activate assistant-axis
set -a; source .env; set +a

mkdir -p slurm/logs/crescendo

export PYTHONPATH="$PROJ_ROOT/src:$PROJ_ROOT/assistant-axis:${PYTHONPATH:-}"

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

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

# Run crescendo attacks with all 9 objectives from PyRIT Gemma-12b batch
python -m crescendo \
    --model "google/gemma-3-12b-it" \
    --objectives-file objectives/harmful_9.json \
    --n-conversations 5 \
    --vectors-dir vectors/ \
    --output-dir experiments/crescendo_runs \
    --max-turns 15 \
    --max-backtracks 5

echo "=== Done ==="
