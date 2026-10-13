#!/bin/bash
#SBATCH --job-name=de430-orbit-rotation-10a
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de430_10a_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de430_10a_%j.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -u -m lunarops run configs/lunarops_orbit_rotation_de430_10a.yml --working-dir .
.venv/bin/python -u -m scripts.compare_rotation_de430 \
  output/rotation_de430_10a_full_20261010 ../data/kernels/de430
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ" > output/rotation_de430_10a_full_20261010/completed
