#!/bin/bash
#SBATCH --job-name=de440-rotation-audit
#SBATCH --partition=all
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_audit_%A_%a.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_de440_audit_%A_%a.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cases=(half_step core_pressure prescribed no_geodetic)
name=${cases[$SLURM_ARRAY_TASK_ID]}
audit=output/rotation_de440_audit_20261011
.venv/bin/python -u -m lunarops run "$audit/$name.yml" --working-dir .
if [[ "$name" != prescribed ]]; then
  .venv/bin/python -u -m scripts.compare_rotation_de430 "$audit/$name" ../data/kernels/de440
fi
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ" > "$audit/$name.completed"
