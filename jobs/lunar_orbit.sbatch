#!/bin/bash

#SBATCH --job-name=lunar-orbit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/lunar_orbit_%j.out
#SBATCH --error=logs/lunar_orbit_%j.err

# Build the DE440 + SB441 mass catalog, then run the lunar orbit
# propagation of configs/lunarops_orbit.yml. Each config is validated
# before it runs.
# Submit from the project directory: sbatch jobs/lunar_orbit.sbatch
# Logs go to logs/ in the submit directory; relative paths in the configs
# resolve against it (--working-dir .).

set -euo pipefail

# The orbit propagator integrates in one process with a fixed-step
# Adams-Bashforth-Moulton integrator; --workers starts CPUs - 1 local worker
# processes that prefetch the ephemeris data of the integration epochs (the
# result is bitwise that of a serial run). A two-sided config
# (backwardDurationSeconds > 0) with parallelArcs keeps one more process busy,
# the backward arc: use CPUs - 2 workers then (N + 2 busy processes).
# Numerical libraries stay single-threaded so the processes use only the
# requested CPUs.
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
workers=$(( SLURM_CPUS_PER_TASK - 1 ))
worker_args=()
if (( workers >= 1 )); then
  worker_args=(--workers "$workers")
fi   # otherwise serial: --workers needs at least one worker
uv run --extra dynamics python -m lunarops run \
  configs/lunarops_orbit.yml \
  --working-dir . ${worker_args[@]+"${worker_args[@]}"}

echo "End time: $(date)"
