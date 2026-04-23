#!/bin/bash
#SBATCH --job-name=axis_qwen_2to5
#SBATCH --partition=scavenger-gpu
#SBATCH --account=chenglab
#SBATCH --gres=gpu:1
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-[23-24],dcc-carlsonlab-gpu-ferc-s-o15-10,dcc-chsi-gpu-ferc-s-i11-1,dcc-gehmlab-gpu-ferc-s-n32-[10-13,23-24],dcc-gehmlab-gpu-ferc-s-z25-[15,17-19],dcc-pearsonlab-gpu-ferc-s-o15-17,dcc-plusds-gpu-ferc-s-j11-[17-18],dcc-plusds-gpu-ferc-s-z25-23,dcc-courses-gpu-[01-10]
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/axis/qwen_%j.out
#SBATCH --error=slurm/logs/axis/qwen_%j.err

# Stages 2-5 of assistant axis pipeline for Qwen 1.8B.
# Stage 1 (response generation) already completed.
# Fixes: use spans from batch_metadata to avoid truncation mismatch.

set -euo pipefail

PROJ_ROOT="/hpc/group/chenglab/wc187/explorations/crescendo-interp"
cd "$PROJ_ROOT"

# Conda
source /hpc/group/chenglab/wc187/miniconda3/etc/profile.d/conda.sh
conda activate assistant-axis

# Environment
set -a; source "$PROJ_ROOT/.env"; set +a
export PYTHONPATH="$PROJ_ROOT/src:$PROJ_ROOT/assistant-axis:${PYTHONPATH:-}"

# Create log directory
mkdir -p slurm/logs/axis

# GPU info
echo "=== GPU INFO ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

MODEL="Qwen/Qwen-1_8B-Chat"
OUTPUT_BASE="outputs/qwen-1_8b-chat"

# ============================================================
# STAGE 2: Extract response activations (fixed span mismatch)
# ============================================================
echo "=========================================="
echo "STAGE 2: Extracting response activations"
echo "=========================================="

# Clear previous incomplete outputs
rm -rf "$OUTPUT_BASE/activations"
rm -rf "$OUTPUT_BASE/scores"

python assistant-axis/pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_BASE/responses" \
    --output_dir "$OUTPUT_BASE/activations" \
    --layers all \
    --batch_size 32 \
    --max_length 1024

echo "Stage 2 complete."
echo "Activation files: $(ls $OUTPUT_BASE/activations/*.pt 2>/dev/null | wc -l)"
echo ""

# ============================================================
# STAGE 3: Judge responses with GPT-4.1-mini
# ============================================================
echo "=========================================="
echo "STAGE 3: Judging responses (GPT-4.1-mini)"
echo "=========================================="

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Cannot continue."
    exit 1
fi

python assistant-axis/pipeline/3_judge.py \
    --responses_dir "$OUTPUT_BASE/responses" \
    --roles_dir assistant-axis/data/roles/instructions \
    --output_dir "$OUTPUT_BASE/scores" \
    --judge_model gpt-4.1-mini \
    --batch_size 10 \
    --requests_per_second 20

echo "Stage 3 complete."
echo ""

# ============================================================
# STAGE 4: Compute per-role vectors (score=3 filtered)
# ============================================================
echo "=========================================="
echo "STAGE 4: Computing per-role vectors"
echo "=========================================="

python assistant-axis/pipeline/4_vectors.py \
    --activations_dir "$OUTPUT_BASE/activations" \
    --scores_dir "$OUTPUT_BASE/scores" \
    --output_dir "$OUTPUT_BASE/vectors" \
    --min_count 50

echo "Stage 4 complete."
echo ""

# ============================================================
# STAGE 5: Aggregate into final axis
# ============================================================
echo "=========================================="
echo "STAGE 5: Computing assistant axis"
echo "=========================================="

python assistant-axis/pipeline/5_axis.py \
    --vectors_dir "$OUTPUT_BASE/vectors" \
    --output "$OUTPUT_BASE/axis.pt"

echo "Stage 5 complete."

# Install into vectors directory
echo ""
echo "Installing axis to vectors/qwen-1_8b-chat/"
mkdir -p vectors/qwen-1_8b-chat

# Backup old axis if exists
if [ -f "vectors/qwen-1_8b-chat/assistant_axis.pt" ]; then
    cp vectors/qwen-1_8b-chat/assistant_axis.pt vectors/qwen-1_8b-chat/assistant_axis_old.pt
    echo "  Backed up old axis to assistant_axis_old.pt"
fi

# Install new axis (wrap in dict for tracker compatibility)
python -c "
import torch
axis = torch.load('$OUTPUT_BASE/axis.pt', map_location='cpu', weights_only=True)
torch.save({'axis': axis}, 'vectors/qwen-1_8b-chat/assistant_axis.pt')
print(f'  Installed axis: shape={axis.shape}, dtype={axis.dtype}')
print(f'  Peak layer: {axis.norm(dim=1).argmax().item()} (norm={axis.norm(dim=1).max():.4f})')
"

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE (Stages 2-5)"
echo "=========================================="
