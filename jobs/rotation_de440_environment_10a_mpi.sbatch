#!/bin/bash
#SBATCH --job-name=de440-env-orbit-rotation-10a-mpi
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_environment_10a_mpi_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_environment_10a_mpi_%j.err

# Run configs/lunarops_orbit_rotation_de440_environment_10a.yml under MPI on one node.
# Submit with: sbatch jobs/rotation_de440_environment_10a_mpi.sbatch
#
# Rank 0 integrates; ranks 1..N-1 prefetch the inputs that depend only on
# time (ephemeris states, frame axes, source-to-source point-mass terms).
# The output matches the serial run bit for bit. Each rank uses one
# thread; the ranks share the kernels through the page cache.
# MPI needs at least 2 ranks (--ntasks >= 2).
#
# Edit before submitting: the project path in --output/--error and cd,
# --ntasks, and output_dir. Each job writes to its own output directory
# and finishes by writing a 'completed' file with the elapsed time.
set -euo pipefail
start=$(date +%s)
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

output_dir="./output/rotation_de440_environment_10a_mpi_${SLURM_JOB_ID}"
mpirun -n "$SLURM_NTASKS" -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS .venv/bin/python -u -m lunarops run \
  configs/lunarops_orbit_rotation_de440_environment_10a.yml \
  --set outputDir="$output_dir" \
  --working-dir . --mpi
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ, elapsed $(( $(date +%s) - start )) s" > "$output_dir/completed"
