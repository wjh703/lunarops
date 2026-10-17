#!/bin/bash

#SBATCH --job-name=lunar-orbit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/lunar_orbit_%j.out
#SBATCH --error=logs/lunar_orbit_%j.err

set -euo pipefail

# The orbit propagator is a single fixed-step ABMD process. Keep numerical
# libraries single-threaded so the allocation matches the Slurm request.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

uv run --extra dynamics python -m lunarops validate configs/lunarops_mass_catalog.yml
uv run --extra dynamics python -m lunarops run configs/lunarops_mass_catalog.yml --working-dir .
uv run --extra dynamics python -m lunarops validate configs/lunarops_orbit.yml
uv run --extra dynamics python -m lunarops run \
  configs/lunarops_orbit.yml \
  --working-dir .

echo "End time: $(date)"
