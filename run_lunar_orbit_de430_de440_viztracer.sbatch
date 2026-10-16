#!/bin/bash
#SBATCH --job-name=lunar-viz-compare
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/lunar_viz_compare_%j.out
#SBATCH --error=/data4/wangjinghuan/deploy_bundle/llrops_project_dev/logs/lunar_viz_compare_%j.err

set -euo pipefail

PROJECT_DIR=/data4/wangjinghuan/deploy_bundle/llrops_project_dev
OUTPUT_ROOT="$PROJECT_DIR/output/de430_de440_viz_compare"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_ROOT"/{de430_baseline,de430_profiled,de440_baseline,de440_profiled}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export UV_HTTP_TIMEOUT=300

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date --iso-8601=seconds)"

uv run --extra dynamics python -m lunarops validate configs/lunarops_orbit_de430.yml --working-dir .
uv run --extra dynamics python -m lunarops validate configs/lunarops_orbit_de440.yml --working-dir .

/usr/bin/time -v -o "$OUTPUT_ROOT/de430_baseline/time.txt" \
  uv run --extra dynamics python -m lunarops run configs/lunarops_orbit_de430.yml \
    --working-dir . \
    --set durationSeconds=31536000.0 \
    --set outputDir="$OUTPUT_ROOT/de430_baseline"

/usr/bin/time -v -o "$OUTPUT_ROOT/de430_profiled/time.txt" \
  uv run --extra dynamics --with viztracer viztracer \
    --output_file "$OUTPUT_ROOT/de430_profiled/trace.json.gz" \
    --tracer_entries 10000000 \
    --min_duration 10us \
    --max_stack_depth 40 \
    --ignore_c_function \
    --ignore_frozen \
    --minimize_memory \
    lunarops/__main__.py run configs/lunarops_orbit_de430.yml \
      --working-dir . \
      --set durationSeconds=31536000.0 \
      --set outputDir="$OUTPUT_ROOT/de430_profiled"

uv run --extra dynamics python scripts/analyze_viztracer.py \
  "$OUTPUT_ROOT/de430_profiled/trace.json.gz" --limit 60 \
  > "$OUTPUT_ROOT/de430_profiled/summary.txt"

/usr/bin/time -v -o "$OUTPUT_ROOT/de440_baseline/time.txt" \
  uv run --extra dynamics python -m lunarops run configs/lunarops_orbit_de440.yml \
    --working-dir . \
    --set durationSeconds=31536000.0 \
    --set outputDir="$OUTPUT_ROOT/de440_baseline"

/usr/bin/time -v -o "$OUTPUT_ROOT/de440_profiled/time.txt" \
  uv run --extra dynamics --with viztracer viztracer \
    --output_file "$OUTPUT_ROOT/de440_profiled/trace.json.gz" \
    --tracer_entries 10000000 \
    --min_duration 10us \
    --max_stack_depth 40 \
    --ignore_c_function \
    --ignore_frozen \
    --minimize_memory \
    lunarops/__main__.py run configs/lunarops_orbit_de440.yml \
      --working-dir . \
      --set durationSeconds=31536000.0 \
      --set outputDir="$OUTPUT_ROOT/de440_profiled"

uv run --extra dynamics python scripts/analyze_viztracer.py \
  "$OUTPUT_ROOT/de440_profiled/trace.json.gz" --limit 60 \
  > "$OUTPUT_ROOT/de440_profiled/summary.txt"

echo "End time: $(date --iso-8601=seconds)"
