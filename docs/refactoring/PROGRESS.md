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

## Phase 4 and verification at d6dc8f6

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

## Requested arithmetic follow-up

- Retained reference-position dtdb arguments and final light-time event reuse
  as requested, with comments explaining their rounding/epoch effects.
- Kept the original weighted normal product. Removed its redundant
  linearization-level symmetrization; solver-side symmetrization remains.
- Isolated inverse-based Cholesky comparison: both two-year adjustments
  converge in 8 iterations. Final parameter differences are at most
  6.22570e-13 of sigma; postfit differences are at most 8.04079e-14 m.
  The original solve implementation remains. Details are in `VALIDATION.md`.
- Final code: 466 tests and Ruff F checks pass. Prediction matches all 721
  baseline epochs. Residual changes reach 1.258122 mm, exceeding the original
  1 micrometre gate; the 16-rank adjustment reaches its iteration limit and
  rejects formal publication. No thresholds or convergence rules changed.
- Follow-up products are in `../refactor_work/accepted_lighttime_final`;
  the earlier all-gates-pass and distribution records apply to d6dc8f6.

## Elegance Phase 0 — baseline stopped at R4 (2026-11-01)

- Source: `dev@ee2b98b`; execution branch: `refactor/elegance`.
- No library, test, production configuration, or numerical changes.
- Checks: 520 tests passed in 8.20 s; Ruff clean; mypy 0 errors
  (209 source files). The mypy ceiling for this round is 0.
- Library `.py/.pyx/.pxi`: 26,026 lines; tests: 9,170 lines.
- Fixed derived configurations: `../refactor_work/elegance_baseline/configs/R1.yml`
  through `R6.yml`. R1/R2 use 30 days; R3 uses
  2019-01-01 to 2019-04-01; R5 uses 2019-01-01 to 2019-07-01
  with four MPI ranks. R4 retains its original prediction dates and step.
  Input and output directory variables point to this workspace and the new baseline.
- R6 is configured for its full ten-year interval with 32 OpenMP threads;
  the installed Cython dynamics extension reports OpenMP enabled.

| Scenario | Status | Wall seconds |
|---|---|---:|
| R1 | passed | 17.911 |
| R2 | passed | 4.470 |
| R3 | passed | 26.036 |
| R4 | FAILED | 5.132 |
| R5 | not run: stop rule | — |
| R6 | not run: stop rule | — |

### Blocking baseline failure

R4 raises `ValueError: EOP UT1-TAI interpolation requested MJD
61300.541667, outside loaded range [37665.000000, 61271.000000].`
The prediction starts at 2026-09-17 21:00 UTC+08, while the supplied
merged EOP ends at 2026-08-19 00:00 UTC. Failure occurs at the first of
4,321 epochs. Full traceback: `../refactor_work/elegance_baseline/R4.log`.

Stopped under ELEGANCE_PLAN §2.10. Phase 0 is incomplete; Phase 1 has
not started. No extrapolation, date change, fallback, or relaxed checks
were introduced. Resumption requires EOP covering the requested prediction
interval, or an explicitly approved change to the R4 baseline interval.
R5 and R6 remain pending.

### R3 profile

494 normal points completed. Wall time includes cProfile overhead and
all seven preprocessing/processing programs; it is not a production
performance measurement. Raw profile: `../refactor_work/elegance_baseline/R3.prof`;
command/timing records: `../refactor_work/elegance_baseline/runs.jsonl`.
Cumulative top 30:

```text
Sun Nov  1 06:52:27 2026    /data4/wangjinghuan/deploy_bundle/refactor_work/elegance_baseline/R3.prof

         36133738 function calls (36126054 primitive calls) in 25.228 seconds

   Ordered by: cumulative time
   List reduced from 3061 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    436/1    0.047    0.000   25.911   25.911 {built-in method builtins.exec}
        7    0.021    0.003   21.212    3.030 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:299(run_program)
      2/1    0.000    0.000   13.407   13.407 <string>:1(<module>)
      2/1    0.000    0.000   13.406   13.406 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    9.256    9.256 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    9.255    9.255 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    9.255    9.255 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:256(main)
        1    0.010    0.010    9.021    9.021 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:49(cmd_run)
        1    0.000    0.000    9.006    9.006 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    8.966    8.966 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:116(rows)
      494    0.009    0.000    8.843    0.018 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.008    0.000    8.747    0.018 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:463(solve)
      494    0.004    0.000    7.332    0.015 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:415(_fixed_point)
     1925    0.032    0.000    7.328    0.004 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:470(update)
        1    0.000    0.000    5.104    5.104 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.018    0.000    4.154    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:251(_station_state_at_utc)
     2416    0.023    0.000    3.984    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:72(displacement_itrf_m)
     1925    0.002    0.000    3.917    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:306(_station_state_from_tdb)
        2    0.115    0.057    3.713    1.857 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:382(_read_iers)
        1    0.000    0.000    3.225    3.225 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:113(read_normal_point_source)
        1    0.140    0.140    3.222    3.222 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:216(parse_mini_file)
        1    0.000    0.000    3.126    3.126 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/earth_orientation.py:20(iers_c04_earth_orientation_parameter)
     2416    0.017    0.000    2.862    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/solid_earth_tide.py:43(displacement_itrf_m)
    35069    0.356    0.000    2.494    0.000 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:146(parse_mini_line)
        1    0.000    0.000    2.487    2.487 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:131(build_observation_processor)
        1    0.000    0.000    2.487    2.487 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:197(build_observation_runtime)
       10    0.000    0.000    2.485    0.248 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/config/context.py:139(create_class)
       10    0.000    0.000    2.484    0.248 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/config/registry.py:189(instantiate)
        1    0.000    0.000    2.471    2.471 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:406(read_iers_c04)
        3    0.460    0.153    2.315    0.772 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/earth_orientation.py:77(read_earth_orientation_parameter)


```
