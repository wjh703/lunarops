#!/bin/bash
#SBATCH --job-name=de430-10y-batchsweep
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_10y_prepare_batch_sweep_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_10y_prepare_batch_sweep_%j.err

set -euo pipefail

cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
mkdir -p logs output/de430_10y_prepare_batch_256 output/de430_10y_prepare_batch_512
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --extra dynamics python -c "from lunarops._dynamics_core import eih_correction; assert callable(eih_correction)"
uv run --extra dynamics python -m lunarops validate configs/lunarops_orbit_de430.yml --working-dir .

for batch_size in 256 512; do
  output_dir="./output/de430_10y_prepare_batch_${batch_size}"
  uv run --extra dynamics python -m lunarops run configs/lunarops_orbit_de430.yml \
    --working-dir . \
    --set durationSeconds=315360000.0 \
    --set preparationBatchSize="${batch_size}" \
    --set outputDir="${output_dir}"
  uv run --extra dynamics python scripts/compare_de430_bsp_orbit.py \
    "${output_dir}/lunar_orbit.dat" \
    ../data/kernels/de430 \
    "${output_dir}/bsp_comparison"
done
