#!/bin/bash
#SBATCH --job-name=de430-lunar-inertia
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --output=llrops_project_dev/logs/de430_lunar_inertia_%j.out
#SBATCH --error=llrops_project_dev/logs/de430_lunar_inertia_%j.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle
mkdir -p llrops_project_dev/logs llrops_project_dev/output/de430_compare
export JULIA_NUM_THREADS=32
export JULIA_NUM_PRECOMPILE_TASKS=4
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

exec julia --project=PlanetaryEphemeris.jl \
  PlanetaryEphemeris.jl/scripts/export_de430_lunar_inertia_baseline.jl \
  /data4/wangjinghuan/deploy_bundle/llrops_project_dev/output/de430_compare/de430_lunar_inertia_j2000
