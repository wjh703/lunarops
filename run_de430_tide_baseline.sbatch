#!/bin/bash
#SBATCH --job-name=de430-tide-baseline
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --output=llrops_project_dev/logs/de430_tide_%j.out
#SBATCH --error=llrops_project_dev/logs/de430_tide_%j.err

set -euo pipefail

cd /data4/wangjinghuan/deploy_bundle
mkdir -p /data4/wangjinghuan/deploy_bundle/llrops_project_dev/output/de430_compare
mkdir -p llrops_project_dev/logs
export JULIA_NUM_THREADS=4
export JULIA_NUM_PRECOMPILE_TASKS=1

exec julia --project=PlanetaryEphemeris.jl \
  PlanetaryEphemeris.jl/scripts/export_de430_tide_baseline.jl \
  /data4/wangjinghuan/deploy_bundle/llrops_project_dev/output/de430_compare/de430_tide_j2000
