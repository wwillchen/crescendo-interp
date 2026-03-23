#!/bin/bash
#SBATCH --job-name=vectors-qwen
#SBATCH --partition=gpu-common
#SBATCH --account=chenglab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/vectors/qwen_%j.out
#SBATCH --error=slurm/logs/vectors/qwen_%j.err

# Qwen 1.8B: ~3.6GB fp16 → 1x any GPU
# Run refusal direction pipeline and compare KL values with reference codebase

set -euo pipefail

PROJ_ROOT="/hpc/group/chenglab/wc187/explorations/crescendo-interp"
cd "$PROJ_ROOT"

source /hpc/group/chenglab/wc187/miniconda3/etc/profile.d/conda.sh
conda activate assistant-axis
set -a; source .env; set +a

mkdir -p slurm/logs/vectors

export PYTHONPATH="$PROJ_ROOT/src:$PROJ_ROOT/assistant-axis:${PYTHONPATH:-}"

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

# Clear previous Qwen run to force fresh computation
rm -rf "$PROJ_ROOT/pipelines/refusal_direction/runs/qwen-1_8b-chat"

# Run the pipeline
# Fix: transformers_stream_generator is installed but broken with transformers 4.57+
# Stub it out before importing Qwen model
echo ""
echo "=== Running refusal direction pipeline for Qwen 1.8B ==="
cd "$PROJ_ROOT/pipelines/refusal_direction"
python -c "
import sys, types
# Replace broken transformers_stream_generator with stub
stub = types.ModuleType('transformers_stream_generator')
stub.init_stream_support = lambda: None
sys.modules['transformers_stream_generator'] = stub

from run_pipeline import run_pipeline
run_pipeline(model_path='Qwen/Qwen-1_8B-Chat')
"

# Print KL comparison
echo ""
echo "=== KL Divergence Comparison ==="
cd "$PROJ_ROOT"
python3 -c "
import json

# Our pipeline results
with open('pipelines/refusal_direction/runs/qwen-1_8b-chat/select_direction/direction_evaluations.json') as f:
    ours = json.load(f)

# Reference results
ref_path = '/hpc/group/chenglab/wc187/explorations/mech-interp/refusal_direction/pipeline/runs/qwen-1_8b-chat/select_direction/direction_evaluations.json'
with open(ref_path) as f:
    ref = json.load(f)

print('=== OUR PIPELINE ===')
ours.sort(key=lambda x: x.get('refusal_score', 999))
print(f'{\"pos\":>5s} {\"layer\":>6s} {\"refusal\":>10s} {\"kl_div\":>10s}')
for e in ours[:15]:
    print(f'{e[\"position\"]:>5d} {e[\"layer\"]:>6d} {e[\"refusal_score\"]:>10.4f} {e[\"kl_div_score\"]:>10.4f}')

print()
print('=== REFERENCE ===')
ref.sort(key=lambda x: x.get('refusal_score', 999))
print(f'{\"pos\":>5s} {\"layer\":>6s} {\"refusal\":>10s} {\"kl_div\":>10s}')
for e in ref[:15]:
    print(f'{e[\"position\"]:>5d} {e[\"layer\"]:>6d} {e[\"refusal_score\"]:>10.4f} {e[\"kl_div_score\"]:>10.4f}')

print()
# Check selected direction
with open('pipelines/refusal_direction/runs/qwen-1_8b-chat/direction_metadata.json') as f:
    our_meta = json.load(f)
print(f'Our selected: pos={our_meta[\"pos\"]}, layer={our_meta[\"layer\"]}')

with open('/hpc/group/chenglab/wc187/explorations/mech-interp/refusal_direction/pipeline/runs/qwen-1_8b-chat/direction_metadata.json') as f:
    ref_meta = json.load(f)
print(f'Ref selected: pos={ref_meta[\"pos\"]}, layer={ref_meta[\"layer\"]}')

# Statistical comparison
import statistics
our_kls = [e['kl_div_score'] for e in ours if e['kl_div_score'] == e['kl_div_score']]
ref_kls = [e['kl_div_score'] for e in ref if e['kl_div_score'] == e['kl_div_score']]
print(f'Our KL: mean={statistics.mean(our_kls):.4f}, median={statistics.median(our_kls):.4f}, min={min(our_kls):.4f}, max={max(our_kls):.4f}')
print(f'Ref KL: mean={statistics.mean(ref_kls):.4f}, median={statistics.median(ref_kls):.4f}, min={min(ref_kls):.4f}, max={max(ref_kls):.4f}')
"

echo ""
echo "=== Done ==="
