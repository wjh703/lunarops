#!/bin/bash
#SBATCH --job-name=orbit-regression
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/regression_orbit_%A_%a.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/regression_orbit_%A_%a.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m scripts.run_regression_suite run --family orbit --task "$SLURM_ARRAY_TASK_ID"
