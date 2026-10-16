# Claude plan execution

The unchanged source plan is `CLAUDE_PLAN.md`. Work continues on
`refactor/review-cleanup`; the original interrupted worktrees are preserved.

## Completed verification

- Phase 0: Claude's baseline at b87c9ca, stored in `../refactor_work/baseline`.
  Claude used a 60-day DE440 arc, six-month residual arc, prediction and a
  two-year INPOP adjustment with 16 MPI ranks. These are the actual recorded
  scenarios; they are shorter than the original plan's one-year orbit.
- Phase 1: ed88f71 and dcb4b76, already completed by Claude.
- Phase 2 config/fileio/base: Claude commit 9adc286 integrated as 692e17f.
- Phase 2 interrupted estimation/observation/dynamics edits recovered and
  callers adapted. 467 tests pass. Four scenario comparisons to the baseline
  pass, with zero numeric differences in the compared products, including
  parameters, covariance, normal equations, restart state and report.
- Phase 2 wall times: orbit 36.9 s; residuals 29.0 s; prediction 15.0 s;
  MPI adjustment 141.6 s. Timings are single runs, not performance claims.
- Phase 2 static checks executed. Ruff F checks pass; full lint and mypy
  still report issues (the baseline also reports lint/type issues). General
  style/type cleanup outside Claude's plan was withdrawn.

## Phase 3

- Registry and observation assembly split; adjustment modules merged; Earth
  orientation modules and frame/scaling services reorganized as specified.
- Shared deformation tensors, shared J2 polynomial, shared lunar degree-2
  field, immutable Kuiper preparation, builder helpers and naming changes.
- 464 tests pass. Full Ruff and mypy executed; inherited findings remain
  (mypy: 54 errors in 19 files). No unrelated broad lint/type cleanup.
- All four scenarios exit successfully. Orbit positions, residuals and
  prediction match the baseline exactly; rotation maximum difference is
  1.505e-36. Adjustment postfit maximum difference is 3.725e-11 m;
  parameter difference is at most 1.306e-12 of its baseline sigma.
  These satisfy the source plan's numerical criteria. Condition-number
  diagnostics and binary checksums differ slightly with roundoff; comparing
  these as if they were physical tolerances would be inappropriate.
- Wall times: orbit 37.9 s; residuals 32.8 s; prediction 17.3 s;
  MPI adjustment 167.2 s (single runs).

## Phase 4 and final verification

- Completed the plan's dynamics, observation cache, solver bookkeeping,
  reporting/IO and MPI changes. Restored `scripts/run_regression_suite.py`
  unchanged from b87c9ca: Phase 1 had removed this explicitly retained script.
  Removed the remaining obsolete EOP duplicate-policy API from Phase 2.
- Four proposed arithmetic changes were tried and withdrawn under the plan's
  numerical gates: reference-only topocentric position, final light-time event
  reuse, square-root-weight normal products and inverse-based Cholesky solves.
  Original arithmetic remains; tolerances were not relaxed. See `VALIDATION.md`.
- 466 tests pass; Ruff F checks and `git diff --check` pass. Full Ruff and mypy
  were run and still report findings (159 lint findings; 45 type errors in
  16 files). Broad style/type cleanup was kept outside this refactor.
- All four recovered Claude scenarios pass the physical numerical criteria.
  Maximum orbit position difference is 0.05244 mm; standalone residuals and
  prediction match exactly. Adjustment converges and the maximum parameter
  difference is 4.927e-14 of its baseline sigma.
- Wall times: orbit 35.1 s; residuals 16.9 s; prediction 9.4 s;
  16-rank MPI adjustment 64.9 s (single runs). Profiles and numerical details
  are recorded in `VALIDATION.md` and `../refactor_work`.
- Wheel and sdist build and distribution verification pass. Clean-wheel CLI
  description, serial prediction and four-rank MPI prediction pass outside
  the source tree, with all 721 prediction rows matching the baseline exactly.
