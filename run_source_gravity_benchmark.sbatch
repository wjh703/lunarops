#!/bin/bash
#SBATCH --job-name=source-gravity-batch
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/source_gravity_batch_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/source_gravity_batch_%j.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
output_dir="./output/source_gravity_batch_${SLURM_JOB_ID}"
.venv/bin/python -c 'from lunarops._dynamics_core import openmp_enabled; assert openmp_enabled()'
OMP_NUM_THREADS=1 .venv/bin/python -u -m scripts.benchmark_source_gravity \
    --output "$output_dir/scalar_control.json" --integration-days 60 --scalar-cache --ring-study
for thread_count in 1 4 8 16 32; do
    OMP_NUM_THREADS="$thread_count" .venv/bin/python -u -m scripts.benchmark_source_gravity \
        --output "$output_dir/threads${thread_count}.json" --integration-days 60
done
