#!/bin/bash
#SBATCH --job-name=de430-10y
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_10y_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_10y_%j.err

set -euo pipefail

cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
mkdir -p logs output/de430_10y
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --extra dynamics python -m lunarops validate configs/lunarops_orbit_de430.yml --working-dir .
uv run --extra dynamics python -m lunarops run configs/lunarops_orbit_de430.yml \
  --working-dir . \
  --set durationSeconds=315360000.0 \
  --set outputDir=./output/de430_10y
