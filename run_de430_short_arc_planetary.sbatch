#!/bin/bash
#SBATCH --job-name=de430-pe-7d
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_short_arc_planetary_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_short_arc_planetary_%j.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle
mkdir -p llrops_project_dev/logs llrops_project_dev/output/de430_short_arc
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export JULIA_NUM_PRECOMPILE_TASKS=4
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

exec julia --project=PlanetaryEphemeris.jl \
  PlanetaryEphemeris.jl/scripts/export_de430_short_arc.jl \
  llrops_project_dev/output/de430_short_arc/planetary_de430_7d.tsv 7.0 3600.0
