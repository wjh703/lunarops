#!/bin/bash
#SBATCH --job-name=de430-rotation-compare
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_rotation_comparison_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_rotation_comparison_%j.err
set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --startup-file=no --project=../PlanetaryEphemeris.jl \
  scripts/export_de430_rotation_comparison.jl output/de430_rotation_comparison/reference 3
.venv/bin/python -m scripts.compare_de430_rotation \
  output/de430_rotation_comparison
