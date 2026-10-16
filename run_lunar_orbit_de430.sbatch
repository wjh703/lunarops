#!/bin/bash
#SBATCH --job-name=lunar-de430
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/lunar_de430_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/lunar_de430_%j.err

set -euo pipefail

PROJECT_DIR=/data4/wangjinghuan/deploy_bundle/llrops_project_dev
CONFIG=configs/lunarops_orbit_de430.yml

cd "$PROJECT_DIR"
mkdir -p logs output/de430

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date --iso-8601=seconds)"

uv run --extra dynamics python -m lunarops validate "$CONFIG" --working-dir .
uv run --extra dynamics python -m lunarops run "$CONFIG" --working-dir .

echo "End time: $(date --iso-8601=seconds)"
