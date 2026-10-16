#!/bin/bash
#SBATCH --job-name=de440-env-rot-10a-5400
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_environment_10a_step5400_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_environment_10a_step5400_%j.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" OMP_WAIT_POLICY=PASSIVE
.venv/bin/python -c 'from lunarops._dynamics_core import openmp_enabled; assert openmp_enabled(), "Rebuild with LUNAROPS_ENABLE_OPENMP=1"'

output_dir="./output/rotation_de440_environment_10a_step5400_openmp_${SLURM_JOB_ID}"
.venv/bin/python -u -m lunarops run \
  configs/lunarops_orbit_rotation_de440_environment_10a.yml \
  --set stepSeconds=5400 \
  --set outputDir="$output_dir" \
  --working-dir .
.venv/bin/python -u -m scripts.compare_rotation_de430 \
  "$output_dir" ../data/kernels/de440
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ" > "$output_dir/completed"
