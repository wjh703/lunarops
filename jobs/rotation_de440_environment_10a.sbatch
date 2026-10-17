#!/bin/bash
#SBATCH --job-name=de440-env-orbit-rotation-10a
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_environment_10a_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_environment_10a_%j.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" OMP_WAIT_POLICY=PASSIVE
.venv/bin/python -c 'from lunarops._dynamics_core import openmp_enabled; assert openmp_enabled(), "Rebuild with LUNAROPS_ENABLE_OPENMP=1"'

output_dir="./output/rotation_de440_environment_10a_openmp_${SLURM_JOB_ID}"
.venv/bin/python -u -m lunarops run \
  configs/lunarops_orbit_rotation_de440_environment_10a.yml \
  --set outputDir="$output_dir" \
  --working-dir .
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ" > "$output_dir/completed"
