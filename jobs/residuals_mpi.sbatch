#!/bin/bash

#SBATCH --job-name=lunarops
#SBATCH --nodes=3
#SBATCH --ntasks=96
#SBATCH --time=24:00:00
#SBATCH --output=logs/lunarops_%j.out
#SBATCH --error=logs/lunarops_%j.err

# Compute LLR O-C residuals (or run a parameter adjustment) under MPI
# across several nodes. Submit from the project directory:
#   sbatch jobs/residuals_mpi.sbatch
# One MPI rank runs per Slurm task (--ntasks, at least 2): rank 0 runs the
# programs, the other ranks evaluate normal points. Logs go to logs/.

set -euo pipefail

# Small LLR normal systems are faster and more stable without per-rank BLAS
# thread pools; MPI provides the forward-model parallelism.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Select the config to run: exactly one line below is active. Uncomment
# another line (and comment out the active one) to run that config.
# uv run --extra mpi mpirun -n "$SLURM_NTASKS" python -m lunarops run configs/lunarops_reflector_bias_adjustment_detailed.yml --mpi
# uv run --extra mpi mpirun -n "$SLURM_NTASKS" python -m lunarops run configs/lunarops_oc_residuals.yml --mpi
# uv run --extra mpi mpirun -n "$SLURM_NTASKS" python -m lunarops run configs/lunarops_oc_residuals_de440.yml --mpi
# uv run --extra mpi mpirun -n "$SLURM_NTASKS" python -m lunarops run configs/lunarops_oc_residuals_epm2021.yml --mpi
# uv run --extra mpi mpirun -n "$SLURM_NTASKS" python -m lunarops run configs/lunarops_oc_residuals_inpop21a.yml --mpi
uv run --extra mpi mpirun -n "$SLURM_NTASKS" python -m lunarops run configs/lunarops_oc_residuals_de440_tdbCompatibleLunarSurface.yml --mpi
# uv run --extra mpi mpirun -n "$SLURM_NTASKS" python -m lunarops run configs/lunarops_oc_residuals_de440_alreadyScaled.yml --mpi
echo "End time: $(date)"
