#!/bin/bash

# Ten-year ABMD step-convergence test. The 5400 s result already exists from
# the production run; this job generates independent 2700 s and 1350 s runs.

#SBATCH --job-name=lunar-orbit-conv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/lunar_orbit_convergence_%j.out
#SBATCH --error=logs/lunar_orbit_convergence_%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/long_arc_convergence

CONFIG=configs/lunarops_orbit.yml
RUNNER=(uv run --extra dynamics python -m lunarops run "$CONFIG" --working-dir .)

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

uv run --extra dynamics python -m lunarops validate "$CONFIG" --working-dir .

run_case() {
  local name="$1"
  local step="$2"
  local output_dir="./output/long_arc_convergence/${name}"
  mkdir -p "$output_dir"
  echo "===== START ${name}: step=${step} s, startup=675 s ====="
  date
  "${RUNNER[@]}" \
    --set "outputDir=${output_dir}" \
    --set "stepSeconds=${step}" \
    --set "startupStepSeconds=675.0"
  echo "===== END ${name} ====="
  date
}

run_case step_2700 2700.0
run_case step_1350 1350.0

echo "All convergence cases completed at $(date)."
