#!/bin/bash
#SBATCH --job-name=axis_qwen_3to5
#SBATCH --partition=scavenger-gpu
#SBATCH --account=chenglab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/axis/qwen_%j.out
#SBATCH --error=slurm/logs/axis/qwen_%j.err

# Resume from Stage 3 (judging). Stages 1-2 already completed.
# The judge script auto-skips already-scored roles (117/276 done).
# Reduced rate limit to avoid rate-limiting issues that killed the previous run.

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
# STAGE 3: Judge responses with GPT-4.1-mini (RESUME)
# Already scored: 117/276 roles. Script auto-skips completed ones.
# Reduced batch_size and requests_per_second to avoid rate limits.
# ============================================================
echo "=========================================="
echo "STAGE 3: Judging responses (GPT-4.1-mini) — RESUMING"
echo "=========================================="

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Cannot continue."
    exit 1
fi

echo "Already scored: $(ls $OUTPUT_BASE/scores/*.json 2>/dev/null | wc -l) roles"
echo "Total roles: $(ls $OUTPUT_BASE/responses/*.jsonl 2>/dev/null | wc -l)"
echo ""

python assistant-axis/pipeline/3_judge.py \
    --responses_dir "$OUTPUT_BASE/responses" \
    --roles_dir assistant-axis/data/roles/instructions \
    --output_dir "$OUTPUT_BASE/scores" \
    --judge_model gpt-4.1-mini \
    --batch_size 10 \
    --requests_per_second 10

echo "Stage 3 complete."
echo "Scored roles: $(ls $OUTPUT_BASE/scores/*.json 2>/dev/null | wc -l)"
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
echo "PIPELINE COMPLETE (Stages 3-5, resumed)"
echo "=========================================="
