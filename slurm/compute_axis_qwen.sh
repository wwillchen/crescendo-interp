#!/bin/bash
#SBATCH --job-name=axis_qwen
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/axis/qwen_%j.out
#SBATCH --error=slurm/logs/axis/qwen_%j.err

# Compute accurate assistant axis for Qwen 1.8B using the full 5-stage pipeline.
# Fixes three flaws in the original compute_mini_axis.py:
#   1. Uses response activations (not prompt activations)
#   2. Filters by GPT-4.1-mini score=3 (not all samples)
#   3. Averages response tokens only (not all tokens)

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
# STAGE 1: Generate responses with vLLM (fast batched inference)
# HuggingFace manual loop is too slow without KV cache for Qwen 1.x.
# vLLM handles KV cache internally. Chat template fix is in generation.py.
# ============================================================
echo "=========================================="
echo "STAGE 1: Generating responses (vLLM)"
echo "=========================================="

python assistant-axis/pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir assistant-axis/data/roles/instructions \
    --questions_file assistant-axis/data/extraction_questions.jsonl \
    --output_dir "$OUTPUT_BASE/responses" \
    --question_count 240 \
    --max_model_len 1024 \
    --max_tokens 256 \
    --temperature 0.7 \
    --tensor_parallel_size 1

echo "Stage 1 complete."
echo ""

# ============================================================
# STAGE 2: Extract response activations
# ============================================================
echo "=========================================="
echo "STAGE 2: Extracting response activations"
echo "=========================================="

python assistant-axis/pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_BASE/responses" \
    --output_dir "$OUTPUT_BASE/activations" \
    --layers all \
    --batch_size 32 \
    --max_length 1024

echo "Stage 2 complete."
echo ""

# ============================================================
# STAGE 3: Judge responses with GPT-4.1-mini
# Requires OPENAI_API_KEY in .env
# ============================================================
echo "=========================================="
echo "STAGE 3: Judging responses (GPT-4.1-mini)"
echo "=========================================="

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set. Skipping Stage 3."
    echo "Run Stage 3 separately after setting the key."
else
    python assistant-axis/pipeline/3_judge.py \
        --responses_dir "$OUTPUT_BASE/responses" \
        --roles_dir assistant-axis/data/roles/instructions \
        --output_dir "$OUTPUT_BASE/scores" \
        --judge_model gpt-4.1-mini \
        --batch_size 50 \
        --requests_per_second 100

    echo "Stage 3 complete."
fi
echo ""

# ============================================================
# STAGE 4: Compute per-role vectors (score=3 filtered)
# ============================================================
echo "=========================================="
echo "STAGE 4: Computing per-role vectors"
echo "=========================================="

if [ -d "$OUTPUT_BASE/scores" ]; then
    python assistant-axis/pipeline/4_vectors.py \
        --activations_dir "$OUTPUT_BASE/activations" \
        --scores_dir "$OUTPUT_BASE/scores" \
        --output_dir "$OUTPUT_BASE/vectors" \
        --min_count 50

    echo "Stage 4 complete."
else
    echo "Skipping: scores directory not found (Stage 3 may have been skipped)."
fi
echo ""

# ============================================================
# STAGE 5: Aggregate into final axis
# ============================================================
echo "=========================================="
echo "STAGE 5: Computing assistant axis"
echo "=========================================="

if [ -d "$OUTPUT_BASE/vectors" ]; then
    python assistant-axis/pipeline/5_axis.py \
        --vectors_dir "$OUTPUT_BASE/vectors" \
        --output "$OUTPUT_BASE/axis.pt"

    echo "Stage 5 complete."

    # Install into vectors directory
    echo ""
    echo "Installing axis to vectors/qwen-1_8b-chat/"

    # Backup old axis
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

else
    echo "Skipping: vectors directory not found."
fi

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
