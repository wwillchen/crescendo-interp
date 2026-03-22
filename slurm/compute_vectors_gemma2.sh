#!/bin/bash
#SBATCH --job-name=vectors-gemma2
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=dcc-h200-gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/vectors/gemma2_%j.out
#SBATCH --error=slurm/logs/vectors/gemma2_%j.err

# Gemma 2 27B: ~54GB bf16 → 1x H200 (141GB)
# Step 1: Compute refusal direction via pipeline
# Step 2: Download assistant axis from HuggingFace (already computed by Lu et al.)

set -euo pipefail

PROJ_ROOT="/hpc/group/chenglab/wc187/explorations/crescendo-interp"
cd "$PROJ_ROOT"

source /hpc/group/chenglab/wc187/miniconda3/etc/profile.d/conda.sh
conda activate assistant-axis
set -a; source .env; set +a

mkdir -p slurm/logs vectors/gemma-2-27b-it

export PYTHONPATH="$PROJ_ROOT/src:$PROJ_ROOT/assistant-axis:${PYTHONPATH:-}"

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

# Step 1: Download assistant axis from HuggingFace (Lu et al. pre-computed)
echo "=== Step 1: Downloading assistant axis from HuggingFace ==="
python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='lu-christina/assistant-axis-vectors',
    filename='gemma-2-27b/assistant_axis.pt',
    repo_type='dataset',
)
shutil.copy(path, 'vectors/gemma-2-27b-it/assistant_axis.pt')
print(f'Downloaded assistant axis to vectors/gemma-2-27b-it/assistant_axis.pt')
"

# Step 2: Compute refusal direction via pipeline
echo ""
echo "=== Step 2: Computing refusal direction ==="
cd "$PROJ_ROOT/pipelines/refusal_direction"
python run_pipeline.py --model_path google/gemma-2-27b-it

# Step 3: Copy refusal artifacts to vectors/
echo ""
echo "=== Step 3: Organizing vectors ==="
RUNS_DIR="$PROJ_ROOT/pipelines/refusal_direction/runs/gemma-2-27b-it"
cp "$RUNS_DIR/direction.pt" "$PROJ_ROOT/vectors/gemma-2-27b-it/refusal_direction.pt"
cp "$RUNS_DIR/direction_metadata.json" "$PROJ_ROOT/vectors/gemma-2-27b-it/metadata.json"
cp "$RUNS_DIR/generate_directions/mean_diffs.pt" "$PROJ_ROOT/vectors/gemma-2-27b-it/refusal_mean_diffs.pt"

echo ""
echo "=== Done ==="
ls -la "$PROJ_ROOT/vectors/gemma-2-27b-it/"
