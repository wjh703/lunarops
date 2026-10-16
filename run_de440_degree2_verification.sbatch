#!/bin/bash
#SBATCH --job-name=de440-degree2
#SBATCH --partition=all
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de440_degree2_%A_%a.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de440_degree2_%A_%a.err
set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cases=(baseline c21 s21 both negative mechanical)
case_name="${cases[$SLURM_ARRAY_TASK_ID]}"
case_dir="output/de440_degree2_verification_20261012/$case_name"
.venv/bin/python -u -m scripts.run_de440_degree2_probe "$case_dir/config.yml" --mode "$case_name"
.venv/bin/python -u -m scripts.compare_rotation_de430 "$case_dir" ../data/kernels/de440
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ" > "$case_dir/completed"
