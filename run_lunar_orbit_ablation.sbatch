#!/bin/bash

# Run the force-model ablation suite used to diagnose the Earth/Moon drift.
# Each case writes to output/ablation/<case> through the shared configuration.

#SBATCH --job-name=lunar-orbit-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/lunar_orbit_ablation_%j.out
#SBATCH --error=logs/lunar_orbit_ablation_%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONFIG=configs/lunarops_orbit_ablation.yml
RUNNER=(uv run --extra dynamics python -m lunarops run "$CONFIG" --working-dir .)

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

uv run --extra dynamics python -m lunarops validate "$CONFIG" --working-dir .

run_case() {
  local name="$1"
  shift
  echo
  echo "===== START CASE: ${name} ====="
  date
  "${RUNNER[@]}" --set "testName=${name}" "$@"
  echo "===== END CASE: ${name} ====="
  date
}

# 1. Complete current model.
run_case baseline

# 2. Remove the 373 small-body sources while retaining DE440 major bodies.
run_case no_sb441 \
  --set 'externalBodyGroups=[de440_major]'

# 3. Remove the EIH term, retaining all Newtonian and non-EIH terms.
run_case no_eih \
  --set 'includeEih=false' \
  --set 'forceTerms=[point_mass,solar_j2,figure_earth,figure_moon,tide,lense_thirring,solar_radiation_pressure]'

# 4. Keep only Newtonian point-mass acceleration.
run_case point_mass_only \
  --set 'includeEih=false' \
  --set 'forceTerms=[point_mass]'

# 5. Remove the Earth/Moon static spherical-harmonic fields.
run_case no_figure \
  --set 'forceTerms=[point_mass,eih_1pn,solar_j2,tide,lense_thirring,solar_radiation_pressure]'

# 6. Remove the delayed/elastic Earth tide term.
run_case no_tide \
  --set 'forceTerms=[point_mass,eih_1pn,solar_j2,figure_earth,figure_moon,lense_thirring,solar_radiation_pressure]'

# 7. Replace the dynamic lunar degree-2 field with the configured static field.
run_case no_lunar_inertia \
  --set 'lunarInertiaConfig=null'

# 8. Remove the solar J2 term.
run_case no_solar_j2 \
  --set 'forceTerms=[point_mass,eih_1pn,figure_earth,figure_moon,tide,lense_thirring,solar_radiation_pressure]'

# 9. Remove solar Lense-Thirring and radiation-pressure terms.
run_case no_solar_relativistic \
  --set 'forceTerms=[point_mass,eih_1pn,solar_j2,figure_earth,figure_moon,tide]'

echo
echo "All ablation cases completed at $(date)."
