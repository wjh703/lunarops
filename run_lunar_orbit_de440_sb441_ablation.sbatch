#!/bin/bash
#SBATCH --job-name=de440-sb441-ablation
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de440_sb441_ablation_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de440_sb441_ablation_%j.err

set -euo pipefail

PROJECT_DIR=/data4/wangjinghuan/deploy_bundle/llrops_project_dev
CONFIG=configs/lunarops_orbit_de440.yml
OUTPUT_ROOT="$PROJECT_DIR/output/de440_sb441_ablation"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_ROOT"/{with_sb441,major_only}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date --iso-8601=seconds)"

uv run --extra dynamics python -m lunarops validate "$CONFIG" --working-dir .

echo "===== with_sb441 ====="
/usr/bin/time -v -o "$OUTPUT_ROOT/with_sb441/time.txt" \
  uv run --extra dynamics python -m lunarops run "$CONFIG" \
    --working-dir . \
    --set durationSeconds=31536000.0 \
    --set outputDir="$OUTPUT_ROOT/with_sb441"

echo "===== major_only ====="
/usr/bin/time -v -o "$OUTPUT_ROOT/major_only/time.txt" \
  uv run --extra dynamics python -m lunarops run "$CONFIG" \
    --working-dir . \
    --set durationSeconds=31536000.0 \
    --set externalBodyGroups='[de440_major]' \
    --set outputDir="$OUTPUT_ROOT/major_only"

echo "End time: $(date --iso-8601=seconds)"
