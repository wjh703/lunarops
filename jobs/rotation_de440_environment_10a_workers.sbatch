#!/bin/bash
#SBATCH --job-name=de440-env-orbit-rotation-10a-workers
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_environment_10a_workers_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_environment_10a_workers_%j.err

# Run configs/lunarops_orbit_rotation_de440_environment_10a.yml with local worker processes on one node.
# Submit with: sbatch jobs/rotation_de440_environment_10a_workers.sbatch
#
# The main process integrates; --workers N worker processes prefetch the
# inputs that depend only on time (ephemeris states, frame axes,
# source-to-source point-mass terms). The output matches the serial run bit
# for bit. Every process uses one thread; the workers share the kernels
# through the page cache. No MPI is needed: this is one Slurm task with
# --cpus-per-task CPUs.
# Each busy process needs a CPU: the main process takes one, so
# workers = CPUs - 1. For a two-sided run (backwardDurationSeconds > 0) the
# main process forks one more process for the backward arc (parallelArcs):
# use workers = CPUs - 2, or pass --set parallelArcs=false.
#
# Edit before submitting: the project path in --output/--error and cd,
# --cpus-per-task, and output_dir. Each job writes to its own output
# directory and finishes by writing a 'completed' file with the elapsed time.
set -euo pipefail
start=$(date +%s)
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

workers=$(( SLURM_CPUS_PER_TASK - 1 ))   # main process (use - 2 for two-sided runs)
output_dir="./output/rotation_de440_environment_10a_workers_${SLURM_JOB_ID}"
.venv/bin/python -u -m lunarops run \
  configs/lunarops_orbit_rotation_de440_environment_10a.yml \
  --set outputDir="$output_dir" \
  --working-dir . --workers "$workers"
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ, elapsed $(( $(date +%s) - start )) s" > "$output_dir/completed"
