#!/bin/bash
#SBATCH --job-name=de440-residual
#SBATCH --partition=all
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de440_residual_%A_%a.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de440_residual_%A_%a.err
set -euo pipefail
cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
root=output/de440_residual_audit_20261012
cases=(control prescribed_orbit effective_quadrupole mantle_x_plus mantle_x_minus mantle_y_plus mantle_y_minus mantle_z_plus mantle_z_minus core_x_plus core_x_minus core_y_plus core_y_minus core_z_plus core_z_minus reference_seed)
.venv/bin/python -u -m scripts.check_de440_rotation_residual run "$root" "${cases[$SLURM_ARRAY_TASK_ID]}"
