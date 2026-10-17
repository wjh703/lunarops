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

## Arithmetic decisions and requested follow-up

Commit d6dc8f6 withdrew all four arithmetic changes after numerical checks.
The user subsequently requested retaining the first two and removing the
extra normal-matrix symmetrization during linearization. The current decisions
are below. No numerical thresholds or convergence rules changed.

| Plan item | Numerical evidence | Current implementation |
| --- | --- | --- |
| Reference station position for topocentric dtdb arguments | Isolated residual change up to 0.359463 mm | Retained as requested, with a rounding-path comment; displaced positions remain in the light path |
| Reuse final light-time events and TT conversions from iteration | Earlier attempt changed residuals up to 89.87 micrometres; current combination with reference-position dtdb changes residuals up to 1.258122 mm and reaches the adjustment iteration limit | Retained as requested, with a comment identifying reuse of events evaluated before the last fixed-point update |
| Form normals using `B = A * sqrt(w)` and `B.T @ B` | Together with the inverse-based solve, the original experiment reached the adjustment iteration limit; postfit differences reached about 1.147 mm | Keep `A.T @ (w[:, None] * A)`; remove its extra `0.5 * (N + N.T)` because the solver symmetrizes for spectral diagnostics and Cholesky |
| Replace four Cholesky-related solves with inverse products | Isolated follow-up converges with both implementations; differences and timings below | Original solve sequence retained; spectrum/diagnostic caching retained |

The isolated fourth-item comparison uses d6dc8f6 source, the original normal
product and the 2016–2018 adjustment with 16 MPI ranks. Both runs converge in
8 adjustment iterations and 80 sigma/weight iterations, with eligible formal
products. Each run also evaluates both solve formulas on the same 15 systems:

- Maximum solution difference on a common system: 1.25733e-14, or
  2.40005e-13 of the original formal sigma.
- Maximum covariance-entry difference: 1.73472e-18; maximum relative
  covariance Frobenius-norm difference: 3.47760e-16.
- Final adjustment parameter difference between the two runs: 3.21132e-14,
  or 6.22570e-13 of original sigma. Maximum current-state residual difference:
  3.21132e-14 m; linearized postfit difference: 8.04079e-14 m.
- Total time for 15 solve calls: original 3.33–3.42 ms, inverse products
  1.61 ms (about 2.1 times faster for these calls). This is a small timing
  sample, not an end-to-end speedup measurement.

Records and the comparison harness are under
`../refactor_work/arithmetic_followup`. The normal product's measured maximum
asymmetry is 1.86265e-9 in absolute matrix units, only 5.58315e-17 relative
to its maximum entry; `NormalEquations` accepts this floating-point asymmetry.

## Current follow-up regression

Products are under `../refactor_work/accepted_lighttime_final`. With the first
two changes retained and linearization symmetrization removed:

- Residuals run successfully, but the maximum baseline difference is
  **0.0012581219302365554 m**, exceeding the original 1 micrometre gate.
- Prediction runs successfully and all parsed fields at 721 epochs match.
- The 16-rank adjustment reaches `MAX_ITERATION_COUNT_REACHED`, exits with
  status 1 and produces diagnostic products only. The original convergence
  and formal-product eligibility rules remain in force.
- `uv run pytest -q`: **466 passed** (16.73 s); Ruff F checks pass.

Thus the current requested light-time arithmetic does not pass all original
physical gates. The validation, performance and distribution records below
describe d6dc8f6, before this follow-up; they are not new measurements of the
current code.

## Numerical regression at d6dc8f6

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
