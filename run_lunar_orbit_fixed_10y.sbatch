#!/bin/bash

# Submit as an array with indices 0 and 1. Both cases use the repaired Earth
# tide implementation and differ only in dynamic lunar degree-2 gravity.

#SBATCH --job-name=lunar10-fixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/lunar10_fixed_%A_%a.out
#SBATCH --error=logs/lunar10_fixed_%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONFIG=configs/lunarops_orbit_ablation.yml
COMMON=(
  --set durationSeconds=315360000.0
)

case "$SLURM_ARRAY_TASK_ID" in
  0)
    CASE_NAME=baseline_fixed_10y
    EXTRA=()
    ;;
  1)
    CASE_NAME=no_lunar_inertia_fixed_10y
    EXTRA=(--set lunarInertiaConfig=null)
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

echo "Job ID: $SLURM_JOB_ID"
echo "Case: $CASE_NAME"
echo "Start time: $(date --iso-8601=seconds)"

uv run --extra dynamics python -m lunarops run "$CONFIG" --working-dir . \
  "${COMMON[@]}" \
  --set "testName=$CASE_NAME" \
  "${EXTRA[@]}"

echo "End time: $(date --iso-8601=seconds)"
