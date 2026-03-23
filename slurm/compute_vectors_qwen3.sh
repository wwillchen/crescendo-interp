#!/bin/bash
#SBATCH --job-name=vectors-qwen3
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=dcc-h200-gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/vectors/qwen3_%j.out
#SBATCH --error=slurm/logs/vectors/qwen3_%j.err

# Qwen 3 32B: ~64GB bf16 → 1x H200 (141GB)
# Step 1: Download assistant axis from HuggingFace (Lu et al. pre-computed)
# Step 2: Compute refusal direction via pipeline

set -euo pipefail

PROJ_ROOT="/hpc/group/chenglab/wc187/explorations/crescendo-interp"
cd "$PROJ_ROOT"

source /hpc/group/chenglab/wc187/miniconda3/etc/profile.d/conda.sh
conda activate assistant-axis
set -a; source .env; set +a

mkdir -p slurm/logs/vectors vectors/Qwen3-32B

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

# Step 1: Download assistant axis from HuggingFace
echo "=== Step 1: Downloading assistant axis from HuggingFace ==="
python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='lu-christina/assistant-axis-vectors',
    filename='qwen-3-32b/assistant_axis.pt',
    repo_type='dataset',
)
shutil.copy(path, 'vectors/Qwen3-32B/assistant_axis.pt')
print(f'Downloaded assistant axis to vectors/Qwen3-32B/assistant_axis.pt')
"

# Step 2: Compute refusal direction via pipeline
echo ""
echo "=== Step 2: Computing refusal direction ==="
cd "$PROJ_ROOT/pipelines/refusal_direction"
python run_pipeline.py --model_path Qwen/Qwen3-32B

# Step 3: Copy refusal artifacts to vectors/
echo ""
echo "=== Step 3: Organizing vectors ==="
RUNS_DIR="$PROJ_ROOT/pipelines/refusal_direction/runs/Qwen3-32B"
cp "$RUNS_DIR/direction.pt" "$PROJ_ROOT/vectors/Qwen3-32B/refusal_direction.pt"
cp "$RUNS_DIR/direction_metadata.json" "$PROJ_ROOT/vectors/Qwen3-32B/metadata.json"
cp "$RUNS_DIR/generate_directions/mean_diffs.pt" "$PROJ_ROOT/vectors/Qwen3-32B/refusal_mean_diffs.pt"

echo ""
echo "=== Done ==="
ls -la "$PROJ_ROOT/vectors/Qwen3-32B/"
