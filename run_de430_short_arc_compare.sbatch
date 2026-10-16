#!/bin/bash
#SBATCH --job-name=de430-compare-7d
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_short_arc_compare_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_short_arc_compare_%j.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev

exec uv run --extra dynamics python scripts/compare_de430_short_arc.py \
  output/de430_short_arc/lunarops/lunar_orbit.dat \
  output/de430_short_arc/planetary_de430_7d.tsv \
  output/de430_short_arc
