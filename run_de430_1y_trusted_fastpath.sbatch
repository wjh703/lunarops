#!/bin/bash
#SBATCH --job-name=de430-1y-trust
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_1y_trusted_fastpath_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_1y_trusted_fastpath_%j.err

set -euo pipefail

cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
mkdir -p logs output/de430_1y_trusted_fastpath
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --extra dynamics python -m lunarops validate configs/lunarops_orbit_de430.yml --working-dir .
/usr/bin/time -v -o output/de430_1y_trusted_fastpath/time.txt \
  uv run --extra dynamics python -m lunarops run configs/lunarops_orbit_de430.yml \
    --working-dir . \
    --set durationSeconds=31536000.0 \
    --set outputDir=./output/de430_1y_trusted_fastpath

cmp output/de430_1y_viztracer/baseline/lunar_orbit.dat \
  output/de430_1y_trusted_fastpath/lunar_orbit.dat
cmp output/de430_1y_viztracer/baseline/lunar_acceleration_diagnostics.dat \
  output/de430_1y_trusted_fastpath/lunar_acceleration_diagnostics.dat

cat output/de430_1y_trusted_fastpath/time.txt
