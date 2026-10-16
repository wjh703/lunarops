#!/bin/bash
#SBATCH --job-name=de430-1y-viz
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_1y_viztracer_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/de430_1y_viztracer_%j.err

set -euo pipefail

cd /data4/wangjinghuan/deploy_bundle/llrops_project_dev
mkdir -p logs output/de430_1y_viztracer/baseline output/de430_1y_viztracer/profiled
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export UV_HTTP_TIMEOUT=300

uv run --extra dynamics python -m lunarops validate configs/lunarops_orbit_de430.yml --working-dir .

/usr/bin/time -v -o output/de430_1y_viztracer/baseline_time.txt \
  uv run --extra dynamics python -m lunarops run configs/lunarops_orbit_de430.yml \
    --working-dir . \
    --set durationSeconds=31536000.0 \
    --set outputDir=./output/de430_1y_viztracer/baseline

/usr/bin/time -v -o output/de430_1y_viztracer/profiled_time.txt \
  uv run --extra dynamics --with viztracer viztracer \
    --output_file output/de430_1y_viztracer/de430_1y.json.gz \
    --tracer_entries 10000000 \
    --min_duration 10us \
    --max_stack_depth 40 \
    --ignore_c_function \
    --ignore_frozen \
    --minimize_memory \
    lunarops/__main__.py run configs/lunarops_orbit_de430.yml \
      --working-dir . \
      --set durationSeconds=31536000.0 \
      --set outputDir=./output/de430_1y_viztracer/profiled

uv run --extra dynamics python scripts/analyze_viztracer.py \
  output/de430_1y_viztracer/de430_1y.json.gz \
  --limit 50 \
  > output/de430_1y_viztracer/summary.txt

ls -lh output/de430_1y_viztracer/de430_1y.json.gz
cat output/de430_1y_viztracer/baseline_time.txt
cat output/de430_1y_viztracer/profiled_time.txt
cat output/de430_1y_viztracer/summary.txt
