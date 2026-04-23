#!/bin/bash
#SBATCH --job-name=vectors-gemma3-12b
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=dcc-h200-gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/vectors/gemma3_12b_%j.out
#SBATCH --error=slurm/logs/vectors/gemma3_12b_%j.err

# Compute refusal direction for google/gemma-3-12b-it.
# Gemma 3 12B text backbone: ~24GB bf16. Llama Guard 2 8B eval model: ~16GB bf16.
# Both fit together on one H200 (141GB) with headroom.
#
# Paper-exact setup (Arditi et al., "Refusal in Language Models Is Mediated
# by a Single Direction"):
#   - mean-diff refusal direction over harmful/harmless pairs
#   - KL+refusal selection on the validation split
#   - jailbreak eval via meta-llama/Meta-Llama-Guard-2-8B (local HF, not API)
#
# The OpenRouter fallback in evaluate_jailbreak.py uses Llama Guard 3, which
# is a different classifier — LLAMAGUARD_USE_LOCAL=1 forces the paper-exact LG2.

set -euo pipefail

PROJ_ROOT="/hpc/group/chenglab/wc187/explorations/crescendo-interp"
cd "$PROJ_ROOT"

source /hpc/group/chenglab/wc187/miniconda3/etc/profile.d/conda.sh
conda activate assistant-axis
set -a; source .env; set +a

mkdir -p slurm/logs/vectors vectors/gemma-3-12b-it

export PYTHONPATH="$PROJ_ROOT/src:$PROJ_ROOT/assistant-axis:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LLAMAGUARD_USE_LOCAL=1  # paper-exact Llama Guard 2 via local HF

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Node: $(hostname) ==="

# Background GPU monitor
gpu_monitor() {
    while true; do
        echo "[GPU] $(date +%H:%M:%S) | $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "Util: %s%% | Memory: %s/%s MiB", $1, $2, $3}')"
        sleep 60
    done
}
gpu_monitor &
GPU_MON_PID=$!
trap "kill $GPU_MON_PID 2>/dev/null" EXIT

echo ""
echo "=== Step 1: Compute refusal direction for google/gemma-3-12b-it ==="
cd "$PROJ_ROOT/pipelines/refusal_direction"
python run_pipeline.py --model_path google/gemma-3-12b-it

echo ""
echo "=== Step 2: Organize vectors into vectors/gemma-3-12b-it/ ==="
RUNS_DIR="$PROJ_ROOT/pipelines/refusal_direction/runs/gemma-3-12b-it"
cp "$RUNS_DIR/direction.pt"                       "$PROJ_ROOT/vectors/gemma-3-12b-it/refusal_direction.pt"
cp "$RUNS_DIR/direction_metadata.json"            "$PROJ_ROOT/vectors/gemma-3-12b-it/metadata.json"
cp "$RUNS_DIR/generate_directions/mean_diffs.pt"  "$PROJ_ROOT/vectors/gemma-3-12b-it/refusal_mean_diffs.pt"

echo ""
echo "=== Done ==="
ls -la "$PROJ_ROOT/vectors/gemma-3-12b-it/"
