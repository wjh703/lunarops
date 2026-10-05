#!/bin/bash
#SBATCH --job-name=de430-10y-pyspy
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_10y_sampling_profile_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_10y_sampling_profile_%j.err

set -euo pipefail

cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
mkdir -p logs output/de430_10y_sampling_profile
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --extra dynamics --with py-spy python -c "from lunarops._dynamics_core import eih_correction; assert callable(eih_correction)"
uv run --extra dynamics --with py-spy python -m lunarops validate configs/lunarops_orbit_de430.yml --working-dir .
uv run --extra dynamics --with py-spy py-spy record \
  --format raw \
  --rate 100 \
  --threads \
  --nonblocking \
  --output output/de430_10y_sampling_profile/de430_10y.raw \
  -- python -m lunarops run configs/lunarops_orbit_de430.yml \
    --working-dir . \
    --set durationSeconds=315360000.0 \
    --set outputDir=./output/de430_10y_sampling_profile
uv run --extra dynamics python scripts/compare_de430_bsp_orbit.py \
  output/de430_10y_sampling_profile/lunar_orbit.dat \
  ../data/kernels/de430 \
  output/de430_10y_sampling_profile/bsp_comparison
