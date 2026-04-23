#!/bin/bash
#SBATCH --job-name=debug-gemma3
#SBATCH --partition=gpu-common
#SBATCH --account=chenglab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/debug_gemma3_%j.out
#SBATCH --error=slurm/logs/debug_gemma3_%j.err
set -euo pipefail
source /hpc/group/chenglab/wc187/miniconda3/etc/profile.d/conda.sh
conda activate assistant-axis
cd /hpc/group/chenglab/wc187/explorations/crescendo-interp
python slurm/_debug_gemma3.py
