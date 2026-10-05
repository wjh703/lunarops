#!/bin/bash
#SBATCH --job-name=de430-ablation
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_hybrid_ablation_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_hybrid_ablation_%j.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
mkdir -p logs output/de430_hybrid
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

run_variant() {
  local output_dir="$1"
  local external="$2"
  local attitude="$3"
  local earth_frame="$4"
  uv run --extra dynamics python -m lunarops run configs/lunarops_orbit_de430_hybrid.yml \
    --working-dir . \
    --set "outputDir=./output/de430_hybrid/${output_dir}" \
    --set "overrideExternalBodies=${external}" \
    --set "overrideLunarAttitude=${attitude}" \
    --set "earthDynamicsFrame=${earth_frame}"
}

run_variant spk_bpc false false de440
run_variant julia_bpc true false de440
run_variant spk_julia false true de440
run_variant julia_julia true true de440
run_variant spk_bpc_de430_earth false false de430Driver
