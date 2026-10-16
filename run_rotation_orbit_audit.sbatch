#!/bin/bash
#SBATCH --job-name=de430-rotation-orbit-audit
#SBATCH --partition=all
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_orbit_audit_%A_%a.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/rotation_orbit_audit_%A_%a.err

set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cases=(coupled_full prescribed_full prescribed_major)
name=${cases[$SLURM_ARRAY_TASK_ID]}
audit=output/rotation_orbit_audit_20261010
.venv/bin/python -u -m lunarops run "$audit/$name.yml" --working-dir .
if [[ "$name" == coupled_full ]]; then
  .venv/bin/python -u -m scripts.compare_rotation_de430 \
    output/rotation_de430_10a_full_20261010 ../data/kernels/de430
fi
date -u +"Completed at %Y-%m-%dT%H:%M:%SZ" > "$audit/$name.completed"
