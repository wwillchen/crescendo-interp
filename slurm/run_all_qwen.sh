#!/bin/bash
# Submit both crescendo and direct runs for Qwen 1.8B
# Usage: bash slurm/run_all_qwen.sh

cd /hpc/group/chenglab/wc187/explorations/crescendo-interp
mkdir -p slurm/logs/crescendo

echo "Submitting multi-objective Qwen 1.8B jobs..."

JOB_CRESC=$(sbatch --parsable slurm/run_multi_objective_qwen.sh crescendo)
echo "  Crescendo (no backtrack): job $JOB_CRESC"

JOB_DIRECT=$(sbatch --parsable slurm/run_multi_objective_qwen.sh direct)
echo "  Direct baseline:         job $JOB_DIRECT"

echo ""
echo "Monitor: squeue -u wc187 -o '%.10i %.14P %.15j %.2t %.10M %.6D %R'"
echo "Logs:    slurm/logs/crescendo/multi_qwen_{$JOB_CRESC,$JOB_DIRECT}.out"
