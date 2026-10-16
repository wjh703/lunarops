# Claude plan: final validation

The source plan is preserved unchanged in `CLAUDE_PLAN.md`. Implementation
continued on `refactor/review-cleanup`, preserving Claude's interrupted
worktrees. Phase 2 recovery is in ee44457 and 3d0a220; Phase 3 is in d331d5b.
The final phase covers the plan's performance changes and verification.

## Scope and retained changes

- Dynamics: evaluate Earth figure only for Moon, patch C20 before its single
  field evaluation, share lunar degree-2 coefficients, return Moon force
  vectors, precompute harmonic normalization, recur longitude harmonics,
  reuse Adams bases, batch Kuiper/output states and use CALCEPH analytic
  angular rates. Removed the finite-difference step configuration.
- Observation: small ephemeris and terrestrial-matrix LRU caches, batched
  potential/Shapiro/Sun-Moon queries, request-local station-state reuse,
  station/coefficient/HARDISP caches, cached file parsing, reduced ERFA warning
  contexts and conditional reflector derivatives.
- Adjustment/IO/MPI: reuse scientific manifests and file hashes, use LibYAML,
  cache rank/condition/update norms, use aligned solver arrays, reuse the
  observation domain, reduce variance-group/CRD/float formatting work,
  broadcast datasets once and send task indices and prediction grid steps.
- Removed the remaining EOP duplicate-policy compatibility API specified in
  Phase 2. Restored `scripts/run_regression_suite.py` unchanged from b87c9ca,
  as the plan explicitly required retaining it.

## Arithmetic changes withdrawn after numerical checks

These proposed Phase 4 optimizations were attempted but are absent from the
final implementation. No numerical thresholds or convergence rules changed.

| Plan item | Observed failure | Final implementation |
| --- | --- | --- |
| Reference station position for topocentric dtdb arguments | Maximum residual change about 0.3595 mm, exceeding the 1 micrometre gate | Full displaced station position |
| Reuse final light-time events and TT conversions from iteration | Maximum residual change about 89.87 micrometres; adjustment convergence also changed | Original final-event recomputation and conversions, with exact-epoch caches |
| Form normals using `B = A * sqrt(w)` and `B.T @ B` | Together with the inverse-based solve below, the two-year adjustment reached its iteration limit; postfit differences reached about 1.147 mm | Original weighted normal-product arithmetic |
| Replace four Cholesky-related solves with inverse products | Same combined adjustment failure; restoring both original operations restored convergence and numerical agreement | Original solve sequence; spectrum/diagnostic caching retained |

## Numerical regression

All comparisons use Claude's b87c9ca products under
`../refactor_work/baseline`. The recovered baseline contains a **60-day**
DE440 orbit, rather than the one-year arc proposed in the written plan. The
original scenarios and scientific inputs were retained; there is no claim
of a newly validated one-year or ten-year integration.

| Scenario | Actual coverage | Final difference from baseline |
| --- | --- | --- |
| DE440 orbit, rotation and Kuiper ring | 60 days, 241 output epochs | Earth position identical; Moon/relative position maximum 5.24434e-5 m, RMS 1.92107e-5 m; maximum relative position difference 1.44956e-13; maximum rotation element difference 3.77476e-15 |
| DE440 residuals | Six months in 2016, 492 normal points | All compared residuals identical |
| Prediction | 721 epochs | All parsed output fields identical |
| INPOP adjustment | 2016–2018, 2,754 normal points, 16 MPI ranks | Converged, formal products eligible; maximum parameter difference 4.92704e-14 of baseline sigma; current-state residual difference 2.55351e-15 m; linearized postfit difference 2.05669e-14 m |

The plan's physical gates are relative orbit error below 1e-10, residual
differences below 1e-6 m and parameter differences below 1e-6 of baseline
sigma. All pass. Binary checksums, path-dependent fingerprints, timings and
condition-number roundoff are not physical acceptance measures. Product
comparisons used the format readers, including the binary orbit tables.

Analytic angular velocity was independently checked against five-point
matrix derivatives with real kernels: maximum differences were 4.315e-15
rad/s for DE440 and 8.904e-17 rad/s for INPOP. Two unit tests also compare
analytic rates to independent matrix finite differences, with and without
the INPOP longitude correction.

Final orbit products are in `../refactor_work/phase4/orbit/out`; the other
three final products are in `../refactor_work/phase4_final/*/out`.

## Performance records

Times below are single wall-clock runs with BLAS/OpenMP threads limited to
one, using the same recovered scenarios. They are observations, not stable
benchmark guarantees; MPI adjustment used 16 ranks.

| Scenario | Phase 3 | Phase 4 |
| --- | ---: | ---: |
| Orbit | 37.9 s | 35.1 s |
| Residuals | 32.8 s | 16.9 s |
| Prediction | 17.3 s | 9.4 s |
| MPI adjustment | 167.2 s | 64.9 s |

Baseline and final cProfile records and top 30 functions by self/cumulative
time are under `../refactor_work/{baseline/profiles,phase4_profiles}`.
Profiled totals (separate from the wall times above): orbit 51.90 to 48.03 s,
residuals 56.42 to 30.62 s and prediction 28.11 to 16.66 s. Claude's baseline
did not contain an MPI-adjustment profile, so no before/after profile claim
is made for that case. The final 16-rank adjustment was also profiled at
rank 0: 75.88 s including worker waits; it converged and passed the same
physical checks. Worker CPU time is not included in that profile.

| Profile counter | Baseline | Final |
| --- | ---: | ---: |
| Orbit `finite_array` | 123,575 | 4,174 |
| Residuals `finite_array` | 946,185 | 141,645 |
| Residuals `xys06a` | 24,624 | 4,939 |
| Residuals HARDISP | 6,894 | 1,148 |
| Prediction `finite_array` | 544,985 | 84,461 |
| Prediction `xys06a` | 11,542 | 2,163 |

## Tests, static checks and distribution

- `uv run pytest -q`: **466 passed** (16.84 s).
- `uv run ruff check --select F lunarops` and `git diff --check`: pass.
- Full Ruff: 159 findings. Full mypy: 45 errors in 16 files. The earlier
  stages also had lint/type findings; these checks are not claimed as clean.
  General style/type repairs were excluded from the requested plan.
- `uv run -m build` and `scripts/verify_distribution.py` on both wheel and
  sdist: pass. The removed normal-equation extension is absent. Artifacts
  are under `../refactor_work/dist_phase4_final`.
- Installed the final wheel with dynamics/MPI dependencies in a separate
  environment, unset `PYTHONPATH`, and ran outside the source tree. Confirmed
  imports resolve to installed `site-packages`.
- `lunarops describe-config`, serial `lunarops run <prediction config>` and
  four-rank `lunarops run <prediction config> --mpi`: pass. Both prediction
  runs reproduce all 721 baseline rows exactly. The plan's `describe` name
  is not an existing CLI command; used its existing `describe-config` command.
- The preserved regression script's `--help` works. Its historical SLURM
  orchestration was not used to replace Claude's recovered four-scenario
  configs; those configs are in `../refactor_work/configs_phase4`. The
  recovered scenario runner and typed numerical comparison are preserved
  as `../refactor_work/tools/run_phase4.sh` and `verify_plan_physics.py`.

Known limitation from the source plan: unnormalized Legendre intermediates
in the harmonic kernel may overflow above roughly degree 150. This is
documented at the kernel; changing the high-degree algorithm is out of scope.
