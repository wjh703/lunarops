#!/bin/bash
#SBATCH --job-name=de440-iau-pole
#SBATCH --partition=all
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de440_iau_pole_%A_%a.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de440_iau_pole_%A_%a.err
set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
root=output/de440_iau2006_comparison_20261012
cases=(A_de440 B_iau2006)
directory="$root/${cases[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python -u -m lunarops run "$directory/config.yml" --working-dir .
.venv/bin/python -u -m scripts.compare_rotation_de430 "$directory" ../data/kernels/de440
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ" > "$directory/completed"
