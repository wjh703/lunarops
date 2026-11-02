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

## Elegance Phase 0 — EOP updated; stopped at R5 (2026-11-01)

- Resumed after the supplied EOP update, retaining all fixed derived configs.
- R4 now succeeds: 4,321 prediction epochs, 27.430 s wall time.
  Log: `../refactor_work/elegance_baseline/R4_resume.log`.
- R5 ran with four MPI ranks over 2019-01-01 through 2019-07-01,
  using the unchanged parametrizations, screening and convergence rules.
  It failed after 78.616 s with `NormalEquationSingularError`:
  `rank=35/37; condition≈inf; obs_count=1026`.
  Failure occurs in the joint estimate's first linearization, after the
  reflector and bias estimate steps. This is an unmodified-code baseline
  failure, not a refactor regression.
  Full traceback: `../refactor_work/elegance_baseline/R5_resume.log`.
- Stopped under ELEGANCE_PLAN §2.10; R6 remains pending. Phase 0 is still
  incomplete, and Phase 1 has not begun. No parameter removal, pseudoinverse,
  threshold changes, or convergence changes were introduced.
- Further work requires resolving the R5 baseline singularity or an approved
  baseline scenario amendment. The plan's fixed-config rule prevents silently
  substituting another arc after this failure.
- No library/test changes; line counts remain 26,026 / 9,170.
- Ruff clean; mypy 0 errors in 209 files. Full pytest rerun: 520 passed in 8.32 s.
- The updated EOP input hashes are saved in
  `../refactor_work/elegance_baseline/updated_eop_sha256.json`.
  Earlier R3 used the previous EOP files; once the baseline blocker is resolved,
  regenerate R3 and its profile using the updated EOP before Phase 1, so a
  changed file extent is not mistaken for a refactor difference.

## Elegance Phase 0 — revised §3.2 baseline runs (2026-11-01)

Executed the user's revised Phase 0 scope on `refactor/elegance`, source
`ee2b98b`; no library, tests, or production configuration changes.
The previous progress commits are `2a3119f` and `0439667`; this entry
records the approved baseline amendment and all requested executions.

### Configuration and provenance

- Renamed the original `configs/R5.yml` to
  `configs/R5_rejected_rankdeficient.yml`; archived its output directory as
  `R5_rejected_rankdeficient/`. Earlier failure logs remain intact.
- Rebuilt R5 from the production estimation config: 2019-01-01 through
  2019-07-01, removing only `tau_F-l_2190d`, `tau_2F-2l_1095d`, and
  `tau_lp_365d` from `empiricalLibration`. Four MPI ranks.
- Built R5L from the same production config: 2012-01-01 through
  2020-01-01, all parameters and unchanged processing steps. Submitted via
  Slurm with 16 MPI ranks, job `44987612`, node66.
- Verified both derived configs against the production config, allowing
  only the specified dates, terms, and data/output directory variables.
- Archived the previous R3 outputs/profile with `_previous_eop` suffixes;
  regenerated R3 with the existing fixed config and updated EOP.
- R6 uses the full 315,576,000 s interval and the existing job's MPI layout:
  one master and eight prefetch workers, one thread per rank. Submitted via
  Slurm as job `44987613`, node13. Both Slurm jobs completed.
- All artifacts, logs, derived configs, launch scripts and accounting records
  are under `../refactor_work/elegance_baseline/`.
  `phase0_input_sha256.json` records config and EOP input hashes;
  `runs.jsonl` records commands, exit codes and wall times. Two initial R3
  launcher attempts failed due to argument ordering before processing began;
  the corrected invocation completed and generated the profile below.

### Scenario results

| Scenario | Result | Wall seconds |
|---|---|---:|
| R1 | passed | 17.911 |
| R2 | passed | 4.470 |
| R3 | passed | 27.741 |
| R4 | passed | 27.430 |
| R5 | FAILED: iteration limit, diagnostic only | 171.776 |
| R5L | passed | 218.785 |
| R6 | passed | 92.013 |

- R1/R2 results and R4's successful post-EOP-update result are retained from
  prior runs. R3 includes cProfile overhead and preprocessing; these are
  single-run timings, not performance claims.
- R3: 494 normal points; all decoded observation rows equal the previous
  EOP baseline exactly (`R3_eop_comparison.json`). The merged EOP now has
  23,650 samples, matching the updated input data.
- R5: final rank 31/31, 1,044 retained observations, 1,025 with nonzero
  final weights. Reflector step reaches 8 iterations without convergence;
  bias converges in 4; joint reaches its 20-iteration limit without
  convergence. `require_formal_product()` raises
  `Adjustment is diagnostic only: ['MAX_ITERATION_COUNT_REACHED']`.
  The process exits 1. Diagnostic report, restart state, residuals and
  normal equations were written; formal solution, covariance and updated
  catalogs were not written. The earlier rank deficiency is gone, but R5
  is not a passing baseline. No iteration limits or tolerances were changed.
- R5L: final rank 44/44; 9,626 input observations, 9,569 retained after
  screening, 9,568 final equations, 9,475 with nonzero final weights.
  Bias converges in 3 iterations; joint converges in 8 iterations with
  80 sigma/weight iterations. The initial reflector step reaches its
  8-iteration limit, but the final joint result converges and is eligible
  for formal publication. Report, restart state, residuals, normals,
  solution, covariance and updated catalogs all exist.
- R6: 3,654 orbit samples and 3,653 diagnostic samples; full ten-year
  time span confirmed. All orbit values are finite. All 31 common orbit
  rows with R1 match bit for bit, maximum absolute difference 0
  (`R6_prefix_check.json`). This checks the serial/MPI common prefix,
  not a future refactor comparison.
- Detailed adjustment summaries: `adjustment_summaries.json`; Slurm states
  and exit codes: `slurm_accounting.txt`.

### Validation and remaining blocker

- 520 tests pass in 9.41 s; Ruff clean; mypy 0 errors in 209 source files.
- Library `.py/.pyx/.pxi`: 26,026 lines; tests: 9,170 lines. Delta: 0 / 0.
- All explicitly requested Phase 0 scenarios have been executed. R5L and
  R6 were run after the R5 failure as expressly requested in this turn;
  no fallback or scene adjustment was made. Phase 0 has **not passed**
  because R5 remains diagnostic only. No Phase 1 changes started.
- Any further R5 scenario or convergence-rule change requires a new explicit
  instruction; the fixed configs have not been altered after these runs.

### Updated R3 cProfile top 30 (cumulative)

```text
Sun Nov  1 07:16:11 2026    /data4/wangjinghuan/deploy_bundle/refactor_work/elegance_baseline/R3.prof

         36157825 function calls (36150141 primitive calls) in 26.929 seconds

   Ordered by: cumulative time
   List reduced from 3061 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    436/1    0.047    0.000   27.627   27.627 {built-in method builtins.exec}
        7    0.022    0.003   21.369    3.053 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:299(run_program)
      2/1    0.000    0.000   15.004   15.004 <string>:1(<module>)
      2/1    0.000    0.000   15.003   15.003 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    9.090    9.090 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    9.088    9.088 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    9.088    9.088 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:256(main)
        1    0.004    0.004    9.052    9.052 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:49(cmd_run)
        1    0.000    0.000    9.046    9.046 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    9.007    9.007 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:116(rows)
      494    0.008    0.000    8.912    0.018 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.008    0.000    8.816    0.018 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:463(solve)
      494    0.004    0.000    7.394    0.015 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:415(_fixed_point)
     1925    0.032    0.000    7.390    0.004 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:470(update)
        1    0.000    0.000    5.114    5.114 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
        1    0.000    0.000    4.241    4.241 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:131(build_observation_processor)
        1    0.000    0.000    4.241    4.241 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:197(build_observation_runtime)
       10    0.000    0.000    4.239    0.424 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/config/context.py:139(create_class)
       10    0.000    0.000    4.238    0.424 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/config/registry.py:189(instantiate)
     2419    0.017    0.000    4.184    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:251(_station_state_at_utc)
     2416    0.023    0.000    4.013    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:72(displacement_itrf_m)
     1925    0.002    0.000    3.952    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:306(_station_state_from_tdb)
        2    0.111    0.056    3.755    1.878 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:382(_read_iers)
        7    0.000    0.000    3.338    0.477 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:87(create_class)
        1    0.000    0.000    3.338    3.338 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:97(_compose_station_displacements)
        5    0.000    0.000    3.338    0.668 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:101(<genexpr>)
        2    0.000    0.000    3.337    1.668 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/builtin_registry.py:18(_coefficient_file)
        1    0.000    0.000    3.335    3.335 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/builtin_registry.py:164(_station_ocean_pole_tide)
        1    0.029    0.029    3.335    3.335 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/ocean_pole_tide.py:79(__init__)
        1    1.198    1.198    3.306    3.306 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/ocean_pole_tide.py:92(_read)


```

## Elegance Phase 0 — second R5 revision; COMPLETE (2026-11-01)

- Implemented the user's second §3.2 baseline amendment. Archived the
  previous config as `configs/R5_rejected_nonconverged.yml`, its outputs
  as `R5_rejected_nonconverged/`, and its log/result with the same prefix.
- New R5 removes `Isigma_l_27.6d` in addition to the previously removed
  three long-period tau terms. `stationPosition.fixedStations` is exactly
  `[APOLLO, WETTZELL]`. Dates, all other parameters, processing steps,
  iteration limits and convergence thresholds remain unchanged.
- Four-rank R5 exits 0 in 150.498 s. Final joint estimate converges in
  16 iterations, 160 sigma/weight iterations; `formalProductEligible: true`.
  Formal solution, covariance and updated station/reflector catalogs exist.
- Final rank 26/26; input/retained observations 1044; final nonzero-weight observation count 1026.
- All Phase 0 scenarios now have passing baselines: R1 17.911 s,
  R2 4.470 s, updated-EOP/profiled R3 27.741 s, updated-EOP R4 27.430 s,
  revised R5 150.498 s, R5L 218.785 s, R6 92.013 s.
  R5L/R6 Slurm and numerical records are in the preceding entry. R3's
  updated cProfile top 30 is also recorded there.
- All baseline artifacts are under `../refactor_work/elegance_baseline/`.
  `phase0_input_sha256.json` now includes the second-revision R5 config;
  `adjustment_summaries.json` now records the successful R5 result.
- Validation immediately before library edits: 520 tests pass in 8.22 s;
  Ruff clean; mypy 0 errors in 209 files. Library/test lines: 26,026/9,170.
- Phase 0 is complete; start Phase 1.1. No library/test changes in Phase 0.

## Elegance Phase 1.1 — array validation narrowed (2026-11-01)

- Phase 0 completion commit: `7c4583d`. Phase 1 has started; this entry
  covers numbered item 1.1 only. Items 1.2–1.7 are not yet implemented.
- Removed unused `rotation_matrix`, removed `parameter_vector` and
  `state_matrix`, and replaced all callers of the latter two with
  `np.asarray(..., float)`. No aliases or compatibility wrappers remain.
- Restricted remaining `finite_array`/`vector3` calls to public constructors:
  replaced their internal uses in IERS solid-tide input conversion,
  terrestrial geometry/transforms, catalog updates, relative-state vector
  conversion and HARDISP result conversion with `np.asarray`.
- Removed the parametrization contract assertion that demanded rejection of
  an invalid internal update length. Physical/IO/regression tests remain.
- Code/test diff: 14 files, 23 insertions, 85 deletions; net −62 lines.
  Library lines: 26,026 → 25,966 (−60); tests: 9,170 → 9,168 (−2).
- Validation: 520 tests pass in 8.13 s; Ruff clean; mypy 0 errors in
  209 source files; `git diff --check` passes. No Cython changes.
- All R1–R5 executions exit 0. `scripts/compare_outputs.py` was invoked
  explicitly with `--rtol 0 --atol 0` and passes for every scenario.
  Native readers additionally confirm byte-identical decoded orbit,
  rotation, acceleration diagnostics, normal matrix N, right-hand side W,
  reference x0, fitted parameters, uncertainties and covariance. Maximum
  numerical difference is 0; R5 joint iteration count remains 16.
- Artifacts/profiles/comparison logs: `../refactor_work/elegance_phase1_1/`.
  Detailed array comparison: `decoded_product_comparison.json`.
  Timings include preprocessing; R3 includes cProfile overhead.

| Scenario | Result | Wall seconds |
|---|---|---:|
| R1 | passed, zero-tolerance comparison | 12.118 |
| R2 | passed, zero-tolerance comparison | 4.449 |
| R3 | passed, zero-tolerance comparison | 25.715 |
| R4 | passed, zero-tolerance comparison | 27.181 |
| R5 | passed, zero-tolerance comparison | 147.990 |

R5L and R6 are not scheduled at Phase 1; successful Phase 0 baselines remain
available. Phase 1 as a whole is still in progress.

### Phase 1.1 R3 cProfile top 20 (cumulative)

```text
Sun Nov  1 07:41:59 2026    /data4/wangjinghuan/deploy_bundle/refactor_work/elegance_phase1_1/R3.prof

         35871164 function calls (35863418 primitive calls) in 24.902 seconds

   Ordered by: cumulative time
   List reduced from 3064 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    436/1    0.050    0.000   25.588   25.588 {built-in method builtins.exec}
        7    0.024    0.003   21.069    3.010 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:299(run_program)
      2/1    0.000    0.000   13.066   13.066 <string>:1(<module>)
      2/1    0.000    0.000   13.065   13.065 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    8.885    8.885 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    8.884    8.884 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    8.884    8.884 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:256(main)
        1    0.004    0.004    8.872    8.872 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:49(cmd_run)
        1    0.000    0.000    8.865    8.865 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    8.827    8.827 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:116(rows)
      494    0.009    0.000    8.731    0.018 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.008    0.000    8.638    0.017 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:463(solve)
      494    0.004    0.000    7.252    0.015 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:415(_fixed_point)
     1925    0.032    0.000    7.248    0.004 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:470(update)
        1    0.000    0.000    5.060    5.060 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.017    0.000    4.055    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:251(_station_state_at_utc)
     2416    0.023    0.000    3.900    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:72(displacement_itrf_m)
     1925    0.002    0.000    3.850    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:306(_station_state_from_tdb)
        2    0.114    0.057    3.735    1.867 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:382(_read_iers)
        1    0.000    0.000    3.175    3.175 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:113(read_normal_point_source)
```

## Elegance Phase 1.2 — internal records trust boundary data

- Removed validation/copies from LightTimeRequest, displacement inputs,
  TroposphereInput, ObservationEquation, NptRecord, LunarRotationState,
  BodyState, EarthOrientationSample, NormalEquations, ProgramSpec and
  ParameterName. IntegratorSettings retains the startup integer-ratio check.
  Removed redundant LunarDynamics and GravityField constructor checks.
- Moved normal-point canonical validation to file import code shared by
  MINI, CRD and native readers. EOP numerical columns are checked together
  by the IERS reader; native EOP already validates parsed numbers.
  Normal-equation reader retains shape/nonnegative-quadratic checks.
- Troposphere models receive leg elevation directly; no dataclasses.replace
  or zenith-elevation field is needed. Removed tests of deleted internal
  validation/immutability contracts and supplied arrays in state test fixtures.
- 507 tests pass in 6.93 s; Ruff clean; mypy 0 errors (210 files).
  R3 all files pass explicit zero-tolerance comparison, maximum difference 0;
  profiled wall time 22.460 s. Artifacts: `../refactor_work/elegance_phase1_2/`.
  Full Phase 1 R1–R5 acceptance remains scheduled after item 1.7.

## Elegance Phase 1.3 — schema owns defaults

- Consumers directly read schema defaults, including MPI chunksize, observation
  options, Earth tides/J2 and solar-relativistic options. Schema fills nested
  mappings; removed repeated dataset/parametrization validation.
- 507 tests pass (6.93 s), Ruff clean, mypy 0 errors (210 files).
  R1 strict comparison passes with maximum difference 0, wall time 12.127 s.
  Artifacts: `../refactor_work/elegance_phase1_3/`.

## Elegance Phase 1.4 — resolve observation times once

- utcOffsetHours range/type/finite checks remain in the numeric schema;
  the shared schema validator checks whole minutes and stores parsed UTC Epoch
  values in resolved config. Prediction consumes those values directly.
  Removed downstream offset validation and the standalone validation API.
- Added schema rejection coverage for out-of-range, fractional-minute,
  non-finite and boolean offsets. Updated the local-time test to assert the
  resolved UTC epoch directly.
- 512 tests pass in 6.90 s; Ruff clean; mypy 0 errors (210 files).
  R4: 4321 epochs, all output files pass explicit zero-tolerance comparison,
  maximum difference 0; wall time 23.616 s.
  Artifacts: `../refactor_work/elegance_phase1_4/`.
- Completed item commits so far: 1.2 `00170a1`, 1.3 `c46b399`,
  1.4 the commit containing this entry. Phase-end R1–R5 acceptance is pending.

## STOP before Phase 1.5–1.7 — plan §2.10 production-consumer audit

Pre-deletion search for item 1.6 contradicts its stated assumptions:

- `config.registry.register` is used by four production parametrizations:
  `reflector_position.py:20`, `station_position.py:20`,
  `station_range_bias.py:112`, `empirical_libration.py:120`.
  They import this decorator from `lunarops.config.registry` and do not all
  register through register_factory as the plan states.
- `classes/builtin_registry.py:170` reads `catalog.info.tidal_model` in
  `_station_ocean_tidal_loading` to reject a configured model/BLQ model mismatch.
  Therefore OceanTidalLoadingCatalogInfo/.info is not test-only; simply
  deleting it would break a production model construction/validation path.

No item 1.6 deletion or replacement architecture has been implemented.
Per §2.10, work stops here; 1.5 and 1.7 remain unstarted. The plan needs to
specify how these production consumers should change before work resumes.
One additional read-only finding for item 1.5: installed ERFA accepts
`dtf2d('TDB', 2020, 1, 1, 0, 0, 0)` and returns JD (2458849.5, 0.0), so
ERFA alone cannot enforce the existing prohibition on civil TDB construction.
Deleting all TDB checks as written would change that contract.

## Elegance Phase 1.5 — scalar epoch simplification (first revision)

- Centralized civil TDB rejection and finite-second checking in _civil_to_jd;
  ERFA rejects invalid civil seconds, while date/seconds splitting retains its
  day-length bound. Kept the revised, uniform TDB error and leap-second rules.
- total_ordering supplies derived comparisons; ERFA calls take functions
  directly; fixed-offset parsing reuses _parse_isot. shifted() uses the internal
  constructor with the identical two-part normalization arithmetic.
- 521 tests pass in 6.74 s; Ruff clean; mypy 0 errors (210 files).
  The saved pre-change implementation agrees bit for bit on 587 shifts/parses
  covering UTC/TT/TDB, day boundaries, leap seconds, positive/negative shifts
  and ISO offsets; derived comparisons agree with tuple ordering.
  Evidence: `../refactor_work/elegance_phase1_5/epoch_compare.log` and
  `verify_epoch.py`. Full R1–R5 acceptance follows item 1.7.
- The revised plan resolves the previous STOP: continue 1.6 using the explicit
  catalog-property migration and defer registry.register to Phase 2.3.

## Elegance Phase 1.6 — dead APIs and displacement wrappers (first revision)

- Removed unused normal-matrix rank/condition helpers, range-bias coverage
  view, ocean-pole grid info, inertia validation/public gravity wrapper,
  UnknownClassError, and OrientationProvider.from_scalar. All callers now use
  KeyError or the OrientationProvider constructor. Internal observation types
  are not exported at package level; dynamics exports match program/test use.
- Removed the displacement result/PolarWobble dataclasses. evaluate() returns
  the same ITRF array; polar wobble contains m1/m2 in arcseconds. Preserved
  independent displacement/torque reference assertions. Inertia-based test
  fixtures use the retained degree-2 physics kernel, without library wrappers.
- BLQ catalog exposes tidal_model directly; configuration/model mismatch and
  file-header checks remain. Deleted the Info-only output fields; retained the
  reader's source path for its file diagnostics and CMC header consistency check.
- Deliberately deferred registry.register and its four production consumers
  to Phase 2.3 as revised. No additional unanticipated consumers were found;
  no new items in the pending-revision list. ObservationAssembly was already
  absent from package exports.
- 518 tests pass in 6.79 s; Ruff clean; mypy 0 errors (211 files).
  R3 passes explicit zero-tolerance comparison, maximum difference 0,
  wall time 22.766 s. Artifacts and consumer audit:
  `../refactor_work/elegance_phase1_6/`.

## Elegance Phase 1.7 — comments/docstrings; Phase 1 acceptance STOP

- Removed implementation-history narratives (v24/v31, previous dependency and
  probe-output history), section banners and docstrings repeating callable names.
  Kept physics, units, symbols, frame/time conventions and archive grammars.
  Executable ASTs, after removing docstrings, are identical in all 65 edited
  files to item 1.6. No Cython source changes.
- 518 tests pass in 6.88 s; Ruff clean; mypy 0 errors (211 files).
- Phase 1 commits: 1.1 `89ac000`, 1.2 `00170a1`, 1.3 `c46b399`,
  1.4 `15efab9`, 1.5 `8092f49`, 1.6 `4ca05d9`, 1.7 this commit.
- Library py/pyx/pxi lines: 26,026 -> 25,036 (−990).
  Test py lines: 9,170 -> 9,066 (−104). These counts exclude documentation.
- Artifacts: `../refactor_work/elegance_phase1/`. Cases use fixed Phase 0
  configs and only override outputDir; comparison explicitly uses
  `--rtol 0 --atol 0`. R3 has 494 NP; R4 has 4321 epochs.

| Case | Wall seconds | Result |
| --- | ---: | --- |
| R1 | 11.948 | PASS; every compared file max_abs=0, max_rel=0 |
| R2 | 4.371 | PASS; every compared file max_abs=0, max_rel=0 |
| R3 | 24.666 | PASS; every compared file max_abs=0, max_rel=0 |
| R4 | 23.082 | PASS; every compared file max_abs=0, max_rel=0 |
| R5 | 10.411 | FAIL before estimation; YAML fingerprint serialization |

R1/R2 decoded native orbit, rotation and acceleration arrays also match
byte for byte (`decoded_dynamics_comparison.json`). R5's normals, solution,
sigma and covariance comparison could not run because no estimate was made.
R5L and R6 are not scheduled for Phase 1; their Phase 0 results remain recorded.

### STOP / pending revision: Epoch serialization in scientific fingerprints

R5 exits 1 with:
`yaml.representer.RepresenterError: ('cannot represent an object', <TimeScale.UTC: 'utc'>)`.
The call chain is build_processing_problem -> _scientific_fingerprint ->
scientific_fingerprint -> yaml.safe_dump. Item 1.4 now stores Epoch in the
resolved startTime/endTime. plain_data converts this dataclass through asdict;
its TimeScale member is a StrEnum and the scalar string branch returns that
same enum rather than a primitive string, which PyYAML SafeDumper rejects.
A direct read-only reproduction with
`yaml.safe_dump(plain_data(Epoch.from_isot('2019-01-01')))` raises the same error.

Per revised §2.10, acceptance failure stops the entire workflow. No serializer
patch, alternative fingerprint, scenario edit or tolerance change was made.
Phase 1 is NOT complete. Next work needs to specify the canonical boundary
serialization of Epoch/TimeScale and verify its fingerprint implications;
then rerun R5 and finish the native parameter/sigma/covariance comparison.
The old production-consumer STOP is resolved by the revised 1.6 instructions;
registry.register remains explicitly deferred to Phase 2.3, not a new blocker.

### R3 cProfile top 20 (cumulative)

```text
Sun Nov  1 08:13:03 2026    ../refactor_work/elegance_phase1/R3.prof

         32341557 function calls (32333799 primitive calls) in 23.979 seconds

   Ordered by: cumulative time
   List reduced from 3051 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    431/1    0.045    0.000   24.548   24.548 {built-in method builtins.exec}
        7    0.023    0.003   18.095    2.585 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:275(run_program)
      2/1    0.000    0.000   14.047   14.047 <string>:1(<module>)
      2/1    0.000    0.000   14.045   14.045 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    7.910    7.910 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    7.908    7.908 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    7.908    7.908 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:245(main)
        1    0.004    0.004    7.896    7.896 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:45(cmd_run)
        1    0.000    0.000    7.889    7.889 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    7.850    7.850 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:115(rows)
      494    0.009    0.000    7.765    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.008    0.000    7.690    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:452(solve)
      494    0.004    0.000    6.403    0.013 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:404(_fixed_point)
     1925    0.031    0.000    6.399    0.003 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:459(update)
        1    0.000    0.000    4.647    4.647 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
        1    0.000    0.000    4.476    4.476 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:127(build_observation_processor)
        1    0.000    0.000    4.476    4.476 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:179(build_observation_runtime)
       10    0.000    0.000    4.474    0.447 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/config/context.py:126(create_class)
       10    0.000    0.000    4.474    0.447 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/config/registry.py:154(instantiate)
        7    0.000    0.000    3.805    0.544 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:84(create_class)
```

## Elegance Phase 1.4a — resolved config stays native data

- Time validators trial-parse the common observation_time_window without
  replacing startTime/endTime. Prediction and dataset builders use that same
  parser; merged the zero/nonzero UTC-offset filtering paths.
- Processing validator now only validates the plan; the builder constructs it
  without storing a dataclass in config. Fingerprint still projects the same
  scientific processing steps. ConfigSchema canonicalizes boundary values to
  native data, including YAML's implicitly parsed dates and Path inputs.
- Added all-production-config coverage for build_run_plan, resolve_defaults,
  recursive native-type assertions and yaml.safe_dump(plain_data(cfg)).
  542 tests pass in 7.71 s; Ruff clean; mypy 0 errors (211 files).
- R1–R5 and the explicit R5 config/manifest fingerprint comparison follow this
  isolated fix commit. Evidence will be appended to the Phase 1 acceptance log.

## Elegance Phase 1 — acceptance PASS after 1.4a

- Isolated fix commit: `6b8fd22`; 542 tests pass, Ruff clean, mypy 0 errors.
  R1–R5 rerun with frozen baseline configs and explicit --rtol 0 --atol 0.
  All compared output files pass. R3 has 494 NP; R4 has 4321 epochs;
  R5 uses 4 ranks, converges in 16 joint iterations, formalProductEligible=true.
- Native binary orbit/rotation/acceleration, R5 N/W/x0/lPl/count, parameter
  values, uncertainties and covariance are bit identical to Phase 0.
  Evidence: `../refactor_work/elegance_phase1_14a/`, including
  decoded_product_comparison.json and fingerprint_comparison.json.

| Case | Wall seconds | Strict comparison |
| --- | ---: | --- |
| R1 | 12.015 | PASS, max_abs=0 / max_rel=0 |
| R2 | 4.366 | PASS, max_abs=0 / max_rel=0 |
| R3 | 22.617 | PASS, max_abs=0 / max_rel=0 |
| R4 | 23.046 | PASS, max_abs=0 / max_rel=0 |
| R5 | 135.326 | PASS, max_abs=0 / max_rel=0 |

### R5 fingerprint comparison

- Phase 0 report: `7ca03426c0e45c8ec566763558965e4edeacfd1a7519c83128ec50edac15038c`.
- Current report: `2e5edbae278bdd4100d748e1d26ef0acce7f333fdfbd72f9c5469a744139c51a`.
- Full hashes differ as expected: manifest codeSha256 changed from
  `05124748eb2925a57eaee283c14bdf089c690a93aa15919e1b6d04be7a4ee1ba`
  to `ef1ceb36c6dc6f32a21d7a92fc5e18d4331aedf996c95b81ae52b5643ca46571`.
  Manifest generated-file paths also reflect the separate output directory.
- Config-only hash (program + models, same frozen baseline paths, manifest
  excluded) is identical on Phase 0 and current source:
  `986c578f942bfded6b698f2da46b137281e106393a6f74bec63103cdca223181`.
  Used the archived Phase 0 source at 7c4583d to compute its payload;
  payload YAML is byte identical. With the Phase 0 manifest fixed, the current
  implementation reproduces the Phase 0 full fingerprint exactly. No hash
  exclusions or fingerprint algorithm changes were added to production code.
- Library/test lines after 1.4a: 25,036 / 9,092; Phase 0: 26,026 / 9,170.
  R5L/R6 were not scheduled for this phase. Previous STOP is resolved;
  Phase 1 is complete and Phase 2 may start.

### R3 cProfile top 20 (cumulative)

```text
Sun Nov  1 08:22:49 2026    ../refactor_work/elegance_phase1_14a/R3.prof

         32350297 function calls (32341928 primitive calls) in 21.926 seconds

   Ordered by: cumulative time
   List reduced from 3052 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    431/1    0.045    0.000   22.498   22.498 {built-in method builtins.exec}
        7    0.026    0.004   18.238    2.605 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:275(run_program)
      2/1    0.000    0.000   11.876   11.876 <string>:1(<module>)
      2/1    0.000    0.000   11.875   11.875 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    7.936    7.936 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    7.934    7.934 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    7.934    7.934 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:245(main)
        1    0.005    0.005    7.922    7.922 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:45(cmd_run)
        1    0.000    0.000    7.915    7.915 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    7.877    7.877 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:115(rows)
      494    0.008    0.000    7.792    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.008    0.000    7.718    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:452(solve)
      494    0.004    0.000    6.430    0.013 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:404(_fixed_point)
     1925    0.030    0.000    6.426    0.003 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:459(update)
        1    0.000    0.000    4.685    4.685 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.017    0.000    3.683    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:240(_station_state_at_utc)
     2416    0.023    0.000    3.567    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:47(displacement_itrf_m)
     1925    0.002    0.000    3.526    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:295(_station_state_from_tdb)
        1    0.000    0.000    3.281    3.281 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.142    0.142    3.278    3.278 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:208(parse_mini_file)
```

## Elegance Phase 2.1 — remove presentation contracts

- Removed UiHints, GUI-only schema flags, configuration presentation/catalog
  modules, field/config/program JSON-schema descriptions and class_json_schema.
  Their uses were confined to description construction, not resolution.
- Removed describe-config and describe-artifacts commands and their description
  trees; describe-program emits only field name, type, default (when declared)
  and description. Updated current CLI documentation and removed obsolete
  presentation assertions while retaining scope/validation assertions.
- 540 tests pass in 7.64 s; Ruff clean; mypy 0 errors (209 files).
  All 24 production configurations resolve byte-identically before/after:
  JSON SHA-256 `34770144d3c0851e9408e3f5304eda0b2aa87a2175dd23ae355ce975da670a13`.
  All 12 built-in describe-program outputs are valid YAML with only the four
  specified field attributes. R3: zero-tolerance pass, maximum difference 0,
  wall time 22.356 s. Artifacts: `../refactor_work/elegance_phase2_1/`.
- Phase 2 has started; item 2.1 is complete. Remaining Phase 2 items and its
  end-of-phase R1–R5 acceptance have not yet been run.

## Elegance Phase 2.2 — simplify field constraints

- Default null acceptance is now false; string fields default to nonempty.
  Existing nullable defaults, optional observation windows/overrides and empty
  catalog labels explicitly retain their declared semantics.
- Replaced numeric bounds with gt/ge/lt/le; shared sequence size validation;
  removed ConfigSchema.validate and moved camelCase enforcement to a recursive
  test over registered schemas.
- 541 tests pass (7.67 s), Ruff clean, mypy 0 errors (209 files).
  All 24 production resolved configurations remain byte identical, SHA-256
  `34770144d3c0851e9408e3f5304eda0b2aa87a2175dd23ae355ce975da670a13`.

## Elegance Phase 2.3 — model-owned schemas and lazy builtins

- Builtin declarations are now a category/type → module:Class table; registry
  loads model modules on demand. Models and all four parametrizations own
  SCHEMA/from_config. Removed registration decorators, eager initialization,
  dual construction paths and _registry_type; parametrizations use TYPE.
- Scalar delay zero models share Zero. Vector displacement zeros retain their
  shapes. Coefficient-file caching is preserved outside the registry.
- Pending plan revision (§2.10): rangeBias/none also has a production consumer
  requiring a structured RangeBiasCorrection with EXPLICIT_ZERO lookup status,
  rather than a scalar/array zero. Retained ZeroRangeBiasModel and moved its
  schema/factory onto the class; no compatibility alias was introduced.
- 541 tests pass (7.77 s), Ruff clean, mypy 0 errors (210 files); 24 resolved
  configurations remain byte identical (same SHA-256 as 2.2).
- Library/test diff: 511 additions / 570 removals, plus 6-line coefficient cache.

## Elegance Phase 2.4 — narrow construction and ownership APIs

- Registry public construction API is register_factory/resolve/create; removed
  normalization/deepcopy and duplicate validation APIs. Shared-schema/listing
  functions live in config/class_configs.py. Observation assembly passes concrete
  ephemeris/frame/EOP dependencies directly; removed its proxy context and both
  create_class factory_context/cache_namespace arguments (also covers 4.6).
- ClassCache owns an ExitStack; contexts close their own caches/transients,
  while MPI owns and closes its borrowed shared cache. All callbacks run on
  cleanup failures, with exceptions propagated; removed failure dictionaries,
  strict cleanup and custom canonicalization. Keys use sorted JSON/default=str;
  observation IDs use itertools.count.
- 541 tests pass (7.78 s), Ruff clean, mypy 0 errors (211 files). All 24 resolved
  production configurations remain byte identical (same hash as 2.2).

## Elegance Phase 2.5 — program declarations and CLI validation

- ProgramSpec is an ordinary frozen dataclass; registry stores spec/callable
  pairs without RegisteredProgram. ArtifactSlot uses native file-header names;
  removed both type translation maps. Input header pre-reading is confined to
  validate_program_artifacts, which is called only by CLI validation.
- Merged resolve-config into `validate --resolve` (no input reads), removed the
  obsolete typed-processing special case, and updated the current CLI guide.
- 541 tests pass (8.43 s), Ruff clean, mypy 0 errors (211 files).

## Elegance Phase 2.6 — schema-driven processing steps

- Added discriminated nested ConfigSchema variants for six processing step types;
  schemas validate step fields, bounds, robust thresholds, selectors and plan
  ordering. Removed hand-written scalar/parser framework and dataclass validation;
  builders construct typed plans without writing objects into resolved config.
- Variance-component constraints run once at the config boundary; removed its
  duplicate from_config/post-init checks and select_blocks selector checks.
- 540 tests pass (7.73 s), Ruff clean, mypy 0 errors (212 files). Removed the
  obsolete direct-constructor validation test; boundary rejection tests remain.
- R5 typed scientific payload is byte identical to Phase 0, config SHA-256
  `986c578f942bfded6b698f2da46b137281e106393a6f74bec63103cdca223181`;
  fixing the baseline manifest reproduces its full fingerprint exactly:
  `7ca03426c0e45c8ec566763558965e4edeacfd1a7519c83128ec50edac15038c`.

## Elegance Phase 2.7 — CLI values and catalog boundaries

- --set values now use yaml.safe_load directly (including YAML's integer/scientific
  notation semantics). Removed per-equation station-ID validation.
- Coordinate schemas live under config; fileio reads/validates coordinate files
  without importing programs. Inline conversion consumes resolved entries in
  programs/catalog_create.py, which combines both catalog programs and replaces
  the ten-argument helper. Duplicate-key checks run at the input boundary.
- Removed internal BLQ coefficient-record/lookup station-ID revalidation.
  Pending revision (§2.10): BLQ source import also consumes validate_station_id;
  retained that file-boundary validation, which the CRD/MINI-only wording did
  not account for. No new fallback or alias was introduced.
- 540 tests pass (7.64 s), Ruff clean, mypy 0 errors (211 files).
  R5 canonical config hash and baseline-manifest fingerprint remain identical
  to Phase 0. Artifacts: ../refactor_work/elegance_phase2/.

## STOP at Elegance Phase 2.8 — incompatible fingerprint instructions (§2.10)

Completed commits: 2.2 `9b74eb4`, 2.3 `e85565d`, 2.4 `91bdfcf`,
2.5 `9d220e6`, 2.6 `c817af5`, 2.7 `406f395`.

The current 2.8 implementation instructions cannot be applied literally while
preserving the Phase 0 fingerprint payload. Production code was not changed
for 2.8; 2.9 is also pending because §2.10 requires stopping the whole flow
when the described implementation is infeasible.

Reproduction: `.venv/bin/python ../refactor_work/elegance_phase2/audit_2_8.py`.
The audit holds the Phase 0 manifest and frozen R5 input/output paths fixed:

- Current typed-plan projection reproduces baseline fingerprint
  `7ca03426c0e45c8ec566763558965e4edeacfd1a7519c83128ec50edac15038c`.
- Removing that projection and hashing resolved config directly yields
  `a76d4d2f20c9e6a1a0732a142c97d1df7e9148dcb38a0353ee4348d2b3b9a49d`.
  The projection encodes typed variance settings/scientific steps, canonical
  defaults and numeric representations, and excludes output steps. The same
  payload must still be constructed somewhere to retain this hash.
- Restricting `_schema_paths` to immediate `kind == "path"` fields loses R1
  manifest resources `program.ephemerides[0].directory`,
  `program.gravityFields.earth.path`, `program.gravityFields.moon.path`.
  Thus the existing schema and nested production configs contradict the
  nonrecursive collection assumption. R5 alone does not expose this loss.
- Audit evidence: `../refactor_work/elegance_phase2/audit_2_8.json`.

Required plan clarification: allow moving the canonical scientific projection
into the unified fingerprint function, and retain nested path discovery (or
replace it with an explicit complete resource declaration). The manifest cache
optimization can then be implemented without changing the scientific payload.
No fallback hashing, restart bypass, baseline replacement or compatibility alias
has been added. Phase 2 is NOT complete.

Line counts (tracked Python files, newline count; 2.1 → 2.7): library 22,181 →
22,040 (−141), tests 9,040 → 9,065 (+25). Including pyx/pxi/pyi, current library
24,522 lines. Per-item committed deltas (including progress/docs):

| Item | Commit | Added | Removed |
| --- | --- | ---: | ---: |
| 2.2 | 9b74eb4 | 790 | 297 |
| 2.3 | e85565d | 533 | 570 |
| 2.4 | 91bdfcf | 226 | 346 |
| 2.5 | 9d220e6 | 149 | 218 |
| 2.6 | c817af5 | 293 | 522 |
| 2.7 | 406f395 | 271 | 319 |


## Verification of completed Phase 2.2–2.7 at the 2.8 STOP

- Final check: 540 tests pass (7.82 s), Ruff clean, mypy 0 errors (211 files).
- Fixed Phase 0 scenarios R1–R5 all pass `--rtol 0 --atol 0`, maximum
  numerical difference 0. R5 uses 4 MPI ranks and the approved Phase 0 config.
- Decoded native products are byte identical: R1/R2 orbit, rotation and
  acceleration tables; R5 N/W/x0, lPl/count/parameter names, solution,
  uncertainty and covariance. R5 converged in 16 joint iterations with
  formalProductEligible=true. Phase 0 processing state remains readable;
  full baseline-manifest fingerprint reproduction is recorded above (this
  does not assert actual restart acceptance across changed source hashes).
- Logs, decoded comparison, fingerprint audit and R3 profile:
  `../refactor_work/elegance_phase2/`. R5L/R6 are not scheduled for Phase 2.

| Scenario | Wall time (s) | Result |
| --- | ---: | --- |
| R1 | 12.387 | max difference 0 |
| R2 | 4.351 | max difference 0 |
| R3 | 24.163 | max difference 0 |
| R4 | 23.129 | max difference 0 |
| R5 | 134.288 | max difference 0 |

R3 cProfile top 20 (cumulative):

```text
Sun Nov  1 08:50:31 2026    ../refactor_work/elegance_phase2/R3.prof

         32331583 function calls (32323559 primitive calls) in 23.475 seconds

   Ordered by: cumulative time
   List reduced from 3041 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    431/1    0.043    0.000   24.047   24.047 {built-in method builtins.exec}
        7    0.025    0.004   18.060    2.580 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000   13.505   13.505 <string>:1(<module>)
      2/1    0.000    0.000   13.505   13.505 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    7.830    7.830 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    7.828    7.828 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    7.828    7.828 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:206(main)
        1    0.000    0.000    7.816    7.816 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    7.809    7.809 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    7.771    7.771 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:115(rows)
      494    0.008    0.000    7.687    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.008    0.000    7.613    0.015 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:452(solve)
      494    0.004    0.000    6.340    0.013 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:404(_fixed_point)
     1925    0.030    0.000    6.335    0.003 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:459(update)
        1    0.000    0.000    4.675    4.675 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
        1    0.000    0.000    3.992    3.992 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:110(build_observation_processor)
        1    0.000    0.000    3.991    3.991 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/assembly.py:161(build_observation_runtime)
       10    0.000    0.000    3.989    0.399 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/config/registry.py:108(create)
     2419    0.017    0.000    3.645    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:240(_station_state_at_utc)
     2416    0.022    0.000    3.530    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:62(displacement_itrf_m)


```

## Elegance Phase 2.2a — compact schema declarations without AST changes

- Compacted 179 schema/helper calls across 20 Python files where they fit within
  120 columns. Only formatting changed; library/tests/scripts AST dumps (without
  source locations) are identical before and after for every Python file.
- Code delta: +179 / −861, net −682 lines. No whole-tree formatter was run.
- 540 tests pass (7.75 s), Ruff clean, mypy 0 errors (211 files).
- Evidence: `../refactor_work/elegance_phase2_finish/2_2a_ast_before.json`,
  `2_2a_ast_result.json`, `2_2a_tests.log`, `2_2a_mypy.log`.

## Elegance Phase 2.8 (first revision) — preserve payload, share digest and file hashes

- Moved processing_scientific_payload and COMPATIBILITY_EXCLUDED_KEYS into
  config/fingerprints.py. processing_fingerprint and model_compatibility_fingerprint
  share _digest; callers contain no payload construction. Retained recursive
  schema path discovery and separate manifests for their distinct resource sets.
- Reused archive.sha256_file's existing 4096-entry LRU: its key is exactly
  (resolved path, size, mtime_ns). All manifest resources, kernels and source
  files already use this function. A boundary test verifies hits plus invalidation
  after same-size/changed-mtime and changed-size/same-mtime rewrites.
- Full four-rank R5 measurements (all hashing occurred on rank 0):

  | Measurement | Before | After |
  | --- | ---: | ---: |
  | sha256_file interface calls | 333 | 333 |
  | Actual file SHA-256 computations (cache misses) | 172 | 172 |
  | Cache hits | 161 | 161 |
  | Interface total time (s) | 0.877554 | 0.772650 |
  | Miss total time (s) | 0.871914 | 0.766793 |
  | Maximum computations for any file path | 1 | 1 |

  The cache was already effective; no call-count reduction is claimed. Timings
  include instrumentation and filesystem-cache variation. Evidence: hashes_*_rank*.json
  and profile_hashes.py under ../refactor_work/elegance_phase2_finish/.
- audit_2_8.py reproduces Phase 0 fingerprint
  `7ca03426c0e45c8ec566763558965e4edeacfd1a7519c83128ec50edac15038c`
  with its fixed manifest. R1 and R5 complete manifest resources are byte
  identical before/after (manifests_before.json/manifests_after.json).
- R5 passes --rtol 0 --atol 0; 16 joint iterations, converged. 541 tests pass
  (7.72 s), Ruff clean, mypy 0 errors (211 files). Previous 2.8 STOP is resolved.
- Library net −7 lines; tests +24; code net +17. This item adds a shared digest
  and an invalidation contract test; the requested cache already existed, so
  there is no redundant hashing implementation to remove. No new module added.

## Elegance Phase 2.9 — dynamics assembly owns model construction

- Moved LunarPropagationProblem and its builder/helpers into dynamics/assembly.py,
  organized around build_forces/build_rotation/build_tides. Default partners,
  diagnostic force groups and solar term names belong to assembly. Program code
  contains orchestration and file output; metadata descriptions are constants.
- Solar flags read resolved schema defaults directly; removed the unused SciPy
  import probe. The optional-dependency test now exercises the actual DOP853
  consumer and still verifies that core program imports work without SciPy.
- Force accumulation order, partner ordering, scientific switches, integrator
  settings and output calculations are unchanged. Removed redundant force-name
  list construction while retaining the same order.
- lunar_orbit.py: 857 → 332 lines. New assembly: 524 lines, replacing 525 net
  program lines, library net −1. Tests net +2: imports follow the owning module
  and the optional-dependency test now supplies a real startup input; code net
  +1 for this architectural split. No compatibility export or new fallback.
- 541 tests pass (7.74 s), Ruff clean, mypy 0 errors (212 files). R1/R2 both pass
  --rtol 0 --atol 0, maximum difference 0. Remaining final scenarios are running.


## Elegance Phase 2 — final acceptance PASS

- Completed 2.2a `9869e79`, revised 2.8 `b0a77ef`, 2.9 `c027e8b`, in that order.
  Phase 2 is complete. The prior 2.8 STOP is resolved by the plan's first revision;
  the individually deferred production consumers recorded under 2.3/2.7 remain
  pending plan revision, as allowed by §2.10. No Phase 3 work was started.
- Final tests: 541 pass (7.61 s); Ruff clean; mypy 0 errors (212 files).
- Commit-level 2.2a AST verification compares all 210 Python files under library,
  tests and scripts between 5baf2cf and 9869e79: all ASTs identical. Evidence:
  2_2a_committed_ast_comparison.json.
- R1–R5 all pass --rtol 0 --atol 0 against the frozen Phase 0 baseline. No
  configuration/physics/tolerance/convergence changes were made for acceptance.

| Scenario | Wall time (s) | Result |
| --- | ---: | --- |
| R1 | 12.369 | max difference 0 |
| R2 | 4.478 | max difference 0 |
| R3 | 22.460 | max difference 0 |
| R4 | 23.487 | max difference 0 |
| R5 | 134.085 | max difference 0 |

- Decoded native products are byte identical: R1/R2 orbit, rotation, acceleration;
  R5 N/W/x0, lPl, observation count, parameter names, solution, uncertainty and
  covariance. R5 uses 4 MPI ranks, converges in 16 joint iterations, and has
  publicationQuality.formalProductEligible=true.
- The Phase 0 restart file decodes successfully. Frozen baseline paths and fixed
  Phase 0 manifest reproduce full fingerprint
  `7ca03426c0e45c8ec566763558965e4edeacfd1a7519c83128ec50edac15038c`.
  Config-only SHA-256 remains
  `986c578f942bfded6b698f2da46b137281e106393a6f74bec63103cdca223181`.
  R1/R5 complete manifest resources are byte identical before/after 2.8/2.9.
- The actual final R5 report fingerprint is
  `b1d0ef8450eb1b5fbe58d50c731faf5c5001136f33c6273d65b4552f412862c7`;
  its codeSha256 is
  `b412fef951133c9f26fc7945ed14b8d4cd1eb122e69da033113265bdb42d0f13`.
  This run has changed source hashes and output paths; the fixed-manifest audit
  above isolates the unchanged scientific payload. No restart fingerprint check
  was bypassed, and actual cross-source-hash restart acceptance is not claimed.

### Phase 2 net lines

Count tracked .py/.pyx/.pxi/.pyi file newlines, excluding documentation, generated
C/C++, outputs and virtual environments. Baseline is Phase 1 final acceptance
`914be7a` (immediately before 2.1); final code is `c027e8b`. Counting complete
file contents avoids Git copy/rename heuristic distortions in diff statistics.

| Scope | Before | After | Net |
| --- | ---: | ---: | ---: |
| Library | 25,159 | 23,844 | −1,315 |
| Tests | 9,092 | 9,079 | −13 |
| Library + tests | 34,251 | 32,923 | −1,328 |
| Scripts | 931 | 953 | +22 |
| All code | 35,182 | 33,876 | **−1,306** |

The library includes 123 unchanged .pyi lines (excluding stubs: 25,036 → 23,721,
net −1,315). The achieved reduction is smaller than the plan's approximate −2k:
model-owned declarations/discriminated schemas retain real production validation,
and 2.9 is a structural move with invariant metadata hoisted, rather than an
algorithm deletion. Item-specific net increases are explained under 2.8/2.9.
Evidence: tracked_line_counts.json; count command uses git ls-tree/git show and
newline counts, with no rename/copy detection.

Artifacts: `../refactor_work/elegance_phase2_finish/` (acceptance.log, *_compare.log,
decoded_product_comparison.json, audit_2_8.json, manifests_*.json, hashes_*_rank*.json,
final_tests.log, final_mypy.log, R3.prof, R3_top20.txt). R5L/R6 were not scheduled.

### Final R3 cProfile top 20 (cumulative)

```text
Sun Nov  1 09:26:30 2026    ../refactor_work/elegance_phase2_finish/R3.prof

         32331543 function calls (32323507 primitive calls) in 21.761 seconds

   Ordered by: cumulative time
   List reduced from 3042 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    432/1    0.043    0.000   22.327   22.327 {built-in method builtins.exec}
        7    0.023    0.003   18.075    2.582 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000   11.842   11.842 <string>:1(<module>)
      2/1    0.000    0.000   11.841   11.841 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    7.909    7.909 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    7.907    7.907 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    7.907    7.907 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:206(main)
        1    0.000    0.000    7.896    7.896 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    7.881    7.881 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    7.841    7.841 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:115(rows)
      494    0.008    0.000    7.756    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.008    0.000    7.681    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:452(solve)
      494    0.004    0.000    6.399    0.013 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:404(_fixed_point)
     1925    0.030    0.000    6.394    0.003 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:459(update)
        1    0.000    0.000    4.645    4.645 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.017    0.000    3.684    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:240(_station_state_at_utc)
     2416    0.024    0.000    3.567    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:62(displacement_itrf_m)
     1925    0.002    0.000    3.534    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:295(_station_state_from_tdb)
        1    0.000    0.000    3.246    3.246 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.141    0.141    3.243    3.243 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:208(parse_mini_file)


```

## Elegance Phase 3 — 3.1 diagnostic gate failed; implementation stopped (2026-11-01)

Starting code: `7fb1774` (Phase 2 accepted). No Phase 3 production change is
retained. Per ELEGANCE_PLAN §2.10, the failed numerical gate stops the entire
flow; 3.1 is incomplete and 3.2–3.5 have not started.

### Experiment and acceptance evidence

Implemented a trial `LunarRhsKernel.terms(y)` with preallocated force/torque,
CMB/geodetic and inertia buffers. Stores captured each contribution immediately
before its existing accumulation; the accumulation expressions and order were
unchanged. Both diagnostic consumers in `programs/lunar_orbit.py` were redirected
to these native terms. Python reference removal was deferred until this gate.
OpenMP native build succeeded. Ran the frozen Phase 0 R1/R2 configurations with
single-thread BLAS/OpenMP, without changing physics, convergence or tolerances.
Both runs exited 0 (timings in `*_phase0_result.json`).

Decoded numeric products were compared directly against Phase 0, including exact
shape and column matching, before any formatting or rounding:

| Product | Max absolute difference | Max component relative difference | Result |
| --- | ---: | ---: | --- |
| R1 orbit | 0 | 0 | Bit identical |
| R2 orbit | 0 | 0 | Bit identical |
| R1 acceleration diagnostics | 2.0679515313825692e-25 m/s² | 2.2888674830530488e-15 | Fails ≤1e-15 |
| R2 acceleration diagnostics | 2.0679515313825692e-25 m/s² | 2.8399002377526144e-15 | Fails ≤1e-15 |
| R1 rotation table | 5.9164567891575885e-31 (mixed diagnostic units) | 2.1288237222474195e-12 | Fails ≤1e-15 |

The R1 integrated quaternion, mantle/core angular velocities and quaternion norm
error are byte-identical. The largest rotation relative difference is row 6,
`figure:EARTH_J2_z_per_s2`: reference `3.8605680065880265e-23`, native
`3.860568006579808e-23`. The diagnostic force maxima occur in the tide z component.
No zero reference component became nonzero.

The failure also holds using the Euclidean norm of each three-component physical
term, so it is not only an elementwise relative-error convention: R1 tide reaches
`1.5485109387758378e-15`; Earth J2 torque reaches `3.7018899579365415e-13`.
R2 tide's maximum vector-relative difference is `3.868356678768927e-16`.

### Disposition and pending revision

Saved the rejected prototype patch, outputs and comparison scripts under
`../refactor_work/elegance_phase3/`, then restored all four changed production
files to `7fb1774` and rebuilt the native extension with OpenMP. No tolerance was
relaxed, no duplicate diagnostic physics or fallback was introduced, and the
untracked user plan was left untouched. Production net lines: **0**.

3.1 needs a revised numerical contract or an explicit strategy that can preserve
both the current native RHS arithmetic and the old Python diagnostic values.
The prescribed capture of existing native intermediate values demonstrably
exceeds the current diagnostic tolerance. This entry does not mark 3.1 or
Phase 3 accepted. R3–R6 were not scheduled after the failed gate.

Evidence: `rejected_3_1_terms.patch`, `prototype_terms.py`, `prototype_build.log`,
`run_case.py`, `R{1,2}_phase0.log`, `R{1,2}_phase0_result.json`,
`prototype_comparison.json`, `prototype_vector_comparison.json`,
`compare_diagnostics.py`, `analyze_vectors.py`, `restored_build.log`.

Restoration verification: OpenMP rebuild succeeded; `pytest tests -q`: **541
passed** (7.79 s); ruff clean; mypy **0 errors**, 212 files. Re-ran R1 (12.13 s)
and R2 (4.33 s) after restoration: all five decoded numeric tables, including
both diagnostics and the rotation table, are byte-identical to Phase 0.
Evidence: `restored_{tests,ruff,mypy}.log`, `restored/comparison.json` and
`restored/R{1,2}_phase0_result.json`. This documentation-only change adds 66 lines;
all production, test and script sources remain unchanged.

## Elegance Phase 3 — 3.1 accepted with revised diagnostic criterion (2026-11-01)

Plan committed separately as `7f62803`, using the user's requested subject.
Resumed from `rejected_3_1_terms.patch`. `force_indices` is now a constructor
argument and stored field of `FusedRhs`. Native preallocated buffers expose force,
torque, CMB, geodetic rate, derivative and inertia diagnostics. Capture branches
leave the existing accumulation expressions/order unchanged: mechanically
removing the added `record_terms` blocks reproduces the original `_eval` and
`_rotation` bodies (`accumulation_audit.json`). Diagnostic arrays are kernel-owned
views, consumed before the next `terms` call; retained test totals are evaluated
before another call overwrites the buffer.

Force/Torque records now contain parameters only. Candidate-independent tide and
degree-2 preparation is selected during compilation. All 13 unsupported returns
raise `ValueError("unsupported: ...")`. Removed `fused_rhs` selection and Python
RHS/diagnostic paths from `LunarDynamics`, evaluation contexts/results, group
physics evaluators and the three native Python-only wrappers. Moved Python
physics to `tests/reference_dynamics.py`, using independent NumPy point-mass,
EIH and tide formulas without those native wrappers. Tests compare the native
RHS and diagnostics to this oracle, preserving their original tolerances. Two
Python field-cache implementation tests were removed; four native diagnostic /
unsupported torque cases were added. Unknown force rejection is also covered.

**Pending revision (§2.10, target skipped):** `_vector.py` remains because
`deformation.py` imports `cross3` (used by the delayed inertia calculation).
That production caller was not listed in 3.1's deletion analysis. Removing it
would require changing the retained deformation implementation; no replacement,
compatibility alias or fallback was introduced. The other deletion targets are
complete.

### Numerical and performance acceptance

Frozen Phase 0 configurations, unchanged physics/tolerances, OpenMP and BLAS
threads = 1. R1 and R2 orbit tables are byte-identical to Phase 0. R1 quaternion,
mantle/core angular velocities and quaternion norm error are byte-identical.

| Diagnostic | Max term / total normalized difference | Max component relative difference (record only) |
| --- | ---: | ---: |
| R1 force decomposition | 3.062466409799028e-23 | 2.2888674830530488e-15 |
| R2 force decomposition | 2.751498260197076e-23 | 2.8399002377526144e-15 |
| R1 torque/CMB/nu/angular acceleration | 7.01508094389822e-16 | 2.1288237222474195e-12 |
| R1 total/mantle inertia and inertia derivative | 0 | 0 |

All normalized criteria pass ≤1e-15. All reference-zero components remain zero.
All rows/terms were checked, with summed reference forces as the force denominator
and reference mantle angular acceleration norm as the rotation denominator.
Inertia uses the maximum absolute tensor component per row. Numeric tables retain
their column names/order and sample counts.

Contemporaneous same-machine comparison used an isolated `7fb1774` worktree,
built with OpenMP, alternating old/new R1 runs three times each. Median wall time:
`7fb1774` **11.795415 s**, 3.1 **11.769285 s**, change **−0.2215%** (passes ≤+3%).
`performance_runs.json` contains all six timings, commands and working directories.

New kernel diagnostic references are frozen under
`../refactor_work/elegance_phase3/diagnostic_reference/R{1,2}/`; subsequent
non-numerical steps compare diagnostics to these byte-for-byte, while orbits
continue to compare to Phase 0. Evidence: `acceptance_3_1.json`, `accepted_3_1/`,
`performance_result.json`, `accumulation_audit.json`, `build_3_1.log`,
`tests_3_1.log`, `ruff_3_1.log`, `mypy_3_1.log`, `lines_3_1.json`.

Validation: OpenMP native build passes; **543 tests pass** (7.65 s); ruff clean;
mypy **0 errors**, 212 source files. Net code lines: library **−578**, tests
**+497**, combined **−81** (oracle migration accounts for the test increase).

## Elegance Phase 3 — 3.2 epoch batch accepted (2026-11-01)

Replaced the camelCase array dictionary, `DynamicsEpochData` records and
`PointMassGravityCache` with one `EpochBatch` of stacked, non-optional arrays.
Unused source gravity/delayed fields contain zeros. Earth C20/J2 polynomials are
computed into batch columns using the same arithmetic; their parameter tuples
travel with the picklable epoch-input recipe. The epoch itself remains the
caller's explicit argument; no per-stage epoch wrapper is allocated.

`kernel.set_epoch(batch, index)` reads array slices directly. Removed
`FusedRhs._load_epoch` and its per-epoch zero-cache allocation. DOP853 prefetch
maps stage offsets to batch/index pairs, and the ABM grid streams batch slices.
SerialEpochSource/MpiEpochSource retain `.stream()`; both return the same batch
type. Updated MPI workers, diagnostic callbacks, the gravity benchmark and all
tests to the batch API; removed scalar/per-record builder APIs entirely.
The scalar seed loop is deliberately retained for 3.5's separate batch change.

Validation: OpenMP build, **543 tests** (8.08 s), ruff and mypy (0 errors, 212
files) pass. R1 (11.84 s) and R2 (4.30 s) orbit tables are byte-identical to
Phase 0; all diagnostics/rotation tables are byte-identical to the frozen 3.1
kernel reference. Actual R1 MPI epoch prefetch with **4 ranks** also produces
all three tables byte-identical to the serial run. Maximum numeric difference
is **0** throughout. Recipe pickle roundtrip and externally streamed batches
remain covered by tests.

Evidence: `../refactor_work/elegance_phase3/accepted_3_2/` (runs and
`exact_comparison.json`), `mpi_3_2/` (4-rank run and `comparison.json`),
`build_3_2.log`, `tests_3_2.log`, `ruff_3_2.log`, `mypy_3_2.log`.

Net lines: library **-102**, tests **-42**, scripts **+0**; code **-144**.

## Elegance Phase 3 — 3.3 history merged (2026-11-01)

Added `DelayedHistory.states(names, epoch)` over accepted nodes and
`EphemerisHistory.states(names, epoch)` for startup. Both reconstruct Moon as
Earth + relative state and use a single query interface. Removed the duplicated
`LunarDynamics.build_history_provider`, `EarthTideModel._delayed_states`, and
`getattr(history, "relative_state", None)` branch. Delayed Earth-tide raisers
now query this interface directly. Consequently the unused delayed prescribed
states were removed from `EpochBatch`; MPI `.stream()` and stacked force source
inputs remain unchanged.

The startup Moon construction changes summation order and is tagged numerical by
3.3. R1/R2 orbit outputs remain byte-identical to Phase 0; their diagnostic
outputs match the frozen 3.1 kernel baseline bit-for-bit. The full R6 completed
in serial with OpenMP 32 threads in **239.823 s** after a local 9-rank MPI attempt
stalled before producing integration output. The 3,654-row R6 orbit table is
byte-identical to Phase 0; maximum Moon position difference is **0 mm**. All
3,654 integrated quaternion/mantle/core spin/norm rows are also bit-identical.
R6 rotation diagnostics differ at most `4.141519752410312e-30`; acceleration
diagnostics differ at most `4.1359030627651384e-25`, consistent with the
existing native diagnostic arithmetic. No tolerance was changed.

Validation: OpenMP build, **544 tests** (7.59 s), ruff clean, mypy **0 errors**
(212 files). The explicit startup-history test exercises Moon reconstruction
and query ordering. Evidence: `r6_3_3/R6_phase3_result.json`,
`r6_comparison.json`, `r6_rotation_state_comparison.json`,
`accepted_3_3/comparison.json`, `build_3_3.log`, `tests_3_3.log`,
`ruff_3_3.log`, `mypy_3_3.log`.

Net lines: library **-54**, tests **+30**, code **-24**.

## Elegance Phase 3 — 3.4 state wrappers removed (2026-11-01)

Removed `MoonRelativeState` and `BarycentricEarthMoonState`. Assembly now forms
the initial six-vector from the two ephemeris rows; orbit output builds Earth,
Moon and EMB vectors directly. It retains the previous floating-point operation
order for Moon = Earth + relative and relative = Moon - Earth, preserving the
sub-micrometer rounding in the serialized relative columns. Replaced the wrapper
roundtrip test with an array relative-state test.

Validation: R1/R2 orbits, both acceleration diagnostics and R1 rotation output
are all byte-identical to their respective Phase 0 orbit / 3.1 diagnostic
references. OpenMP build remains valid; **544 tests** (7.65 s), ruff clean,
mypy **0 errors** (211 source files). Evidence: `accepted_3_4/`,
`tests_3_4.log`, `ruff_3_4.log`, `mypy_3_4.log`.

Net lines: library **-86**, tests **-3**, code **-89**.

## Elegance Phase 3 — 3.5 integrator startup/main loop merged (2026-11-01)

Both integration paths now use `_main_grid` for regular output steps. The
non-delayed path supplies its backward DOP853 derivative seed and begins ABM at
the first output step; delayed propagation supplies its explicit rigid seed and
uses DOP853 for the same first `order - 1` forward steps before ABM. The common
loop owns epoch-batch streaming, startup stage prefetch, accepted derivatives,
and node diagnostics. Delayed rigid-seed epochs are built as one `EpochBatch`
in their existing time order instead of one batch per epoch. ABM remains Python.

Validation: **544 tests** (7.57 s), ruff clean, mypy 0 errors (211 files).
R1/R2 orbit and diagnostic tables are byte-identical to Phase 0 / frozen 3.1
references. Evidence: `accepted_3_5/`, `tests_3_5.log`, `ruff_3_5.log`,
`mypy_3_5.log`. Phase-end R6 remains running/pending.

Net lines: library **+5**, tests **+0**, code **+5**.

## Elegance Phase 3 — final acceptance (2026-11-01)

Phase 3 items 3.1–3.5 have been implemented and accepted. The single skipped
3.1 deletion target is `_vector.py`: `deformation.py` is still a production
caller, as recorded under §2.10. No compatibility path was added. Phase 4 has
not started.

Commits:

- `7f62803`: separately commit the user-approved 3.1 diagnostic criterion.
- `32b8314`: sole native production RHS, diagnostic term buffers and test oracle.
- `89a038a`: stacked EpochBatch and batch/index integration/MPI interfaces.
- `3e76071`: shared startup/delayed history query interface.
- `89365a2`: remove Earth/Moon state wrappers, preserve output arithmetic order.
- `3dafd72`: shared regular-step integration loop and batched rigid seed epochs.

### Final regression results

All runs used the unchanged frozen Phase 0 configurations. R1/R2 orbit tables
compare to Phase 0 byte-for-byte; their diagnostic tables compare to the accepted
3.1 native diagnostic reference byte-for-byte. R3/R4/R5 use the repository's
zero-tolerance directory comparator, plus decoded R5 product comparisons.

| Case | Wall time, seconds | Result |
| --- | ---: | --- |
| R1 | 12.368 | Orbit, integrated rotation, native diagnostics bit identical |
| R2 | 4.597 | Orbit and native diagnostics bit identical |
| R3 (cProfile enabled) | 22.493 | Zero-tolerance output comparison passes |
| R4 | 22.814 | Zero-tolerance output comparison passes |
| R5 (4 MPI ranks) | 133.966 | Zero-tolerance comparison; decoded parameters, sigma, normals and covariance bit identical |
| R6 (10 years, OpenMP 32 threads) | 236.230 | Complete orbit and integrated rotation state bit identical to Phase 0 |

R5 retains 16 joint adjustment iterations and converged formal output. R5L is
not scheduled in Phase 3. R6 contains 3,654 orbit/rotation samples and 3,653
acceleration diagnostic samples. Its maximum Moon position difference is
**0 mm**, below the 0.01 mm limit. Every integrated quaternion, mantle/core
angular velocity and quaternion norm error is byte-identical to Phase 0.
R6 diagnostic differences from the former Python Phase 0 diagnostics are at
most 4.1359030627651384e-25 m/s² (force table) and 4.141519752410312e-30
(rotation table, mixed diagnostic units). Its common 30-day diagnostic prefix
is byte-identical to the 3.1 native reference; those diagnostic arithmetic
changes were accepted in 3.1 and are not new integration differences.

3.1's same-machine three-run median performance gate remains passed:
11.769285 s versus 7fb1774's 11.795415 s, **−0.2215%**. Final individual R1
timing above is a separate verification run, not a replacement for that median.

Final validation: OpenMP extension rebuilt during 3.2; remaining edits are
Python only. `pytest tests -q`: **544 passed in 7.52 s**; ruff clean; mypy
**0 errors** in 211 source files. Working tree was clean before this final
progress update.

### Net code lines

Complete tracked .py/.pyx/.pxi/.pyi file newline counts, excluding documentation,
generated C/build outputs and environments; before `7fb1774`, after `3dafd72`.

| Scope | Before | After | Net |
| --- | ---: | ---: | ---: |
| Library | 23,844 | 23,029 | **−815** |
| Tests | 9,079 | 9,561 | +482 |
| Scripts | 953 | 953 | 0 |
| All code | 33,876 | 33,543 | **−333** |

Reduction is smaller than the approximate −1.3k target: independent NumPy
physics lives in tests, the delayed/node storage and parameter objects remain
necessary, and `_vector.py` retains its deformation caller. Item 3.5 adds five
library lines: the extracted common loop requires explicit arguments/call sites,
and batching the seed requires an explicit batch and row index; it removes the
duplicate main loop while leaving the two physically distinct seed algorithms.
The historical 3.4 phrase "sub-micrometer rounding" should read "micrometer-scale
rounding": serializing Moon-minus-Earth after barycentric reconstruction can
differ from the integration vector by about 14 micrometers. The original
reconstruction/subtraction order was preserved and final outputs are exact.

Per-item actual Git stats (including progress documentation): 3.1 +1014/−1030
(net −16), 3.2 +237/−350 (net −113), 3.3 +110/−103 (net +7), 3.4 +44/−116
(net −72), 3.5 +47/−25 (net +22). File-copy detection affects gross additions
and deletions for moved physics, but not the net totals.

Artifacts: `../refactor_work/elegance_phase3_finish/` contains final R1–R6
outputs, acceptance.log, R3–R5 comparison logs, decoded_product_comparison.json,
R6_comparison.json, tracked_line_counts.json, final_tests.log, final_ruff.log,
final_mypy.log, R3.prof, R3_top20.txt and run timing JSON. The final R6 was
rerun after 3.5; its original runner reused the `r6_3_3/` path and prototype
attempt label. A copy of its final outputs/log/result is frozen in this final
acceptance directory. The earlier 239.823 s R6 run was the 3.3 checkpoint.

### Final R3 cProfile top 20 (cumulative)

```text
Sun Nov  1 12:04:35 2026    ../refactor_work/elegance_phase3_finish/R3.prof

         32328282 function calls (32320254 primitive calls) in 21.814 seconds

   Ordered by: cumulative time
   List reduced from 3036 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    425/1    0.043    0.000   22.380   22.380 {built-in method builtins.exec}
        7    0.025    0.004   18.136    2.591 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000   11.818   11.818 <string>:1(<module>)
      2/1    0.000    0.000   11.817   11.817 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    7.882    7.882 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    7.881    7.881 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    7.881    7.881 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:206(main)
        1    0.000    0.000    7.869    7.869 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    7.862    7.862 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    7.824    7.824 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:115(rows)
      494    0.008    0.000    7.739    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.008    0.000    7.665    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:452(solve)
      494    0.004    0.000    6.377    0.013 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:404(_fixed_point)
     1925    0.030    0.000    6.373    0.003 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:459(update)
        1    0.000    0.000    4.674    4.674 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.018    0.000    3.671    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:240(_station_state_at_utc)
     2416    0.022    0.000    3.554    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:62(displacement_itrf_m)
     1925    0.002    0.000    3.517    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:295(_station_state_from_tdb)
        1    0.000    0.000    3.235    3.235 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.142    0.142    3.230    3.230 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:208(parse_mini_file)


```

## Elegance Phase 4.1 — unified light-time solution (2026-11-01)

- Replaced `solve_uplink`/`solve` with `solve(request, *, two_way=True)`;
  prediction explicitly uses `two_way=False`. Both legs share `_leg`.
- Removed closure/tuple fixed-point iteration and both iteration-state records.
  The explicit loop retains 12 evaluations, the strict 1e-12 s convergence
  test, expression order, and the distinct one-way/two-way final event epochs.
  Only LightTimeRequest, UplinkSolution and LightTimeSolution remain dataclasses;
  leg/station event values use immutable named tuples. All callers/exports/tests
  use the new API with no aliases or fallback.
- Compared every solution field against 354575f for both paths with iteration
  budgets 1, 2 and 12: identical, including exhausted/nonconverged iterations.
  This audit changes iteration budgets only in its isolated verification process.
  Prediction's test now rejects any receive-event evaluation.
- Full checks: **544 passed in 7.65 s**, ruff clean, mypy 0 errors (211 files).
  No Cython changes; OpenMP rebuild not required.
- R1/R2 orbit and integrated rotation match Phase 0 byte-for-byte; diagnostics
  match the frozen Phase 3.1 native reference byte-for-byte. R3/R4/R5 pass the
  zero-tolerance directory comparator. R5 (4 ranks) decoded parameters, sigma,
  N/W/x0/lPl, observation count and covariance match exactly; 16 joint iterations,
  converged. Maximum R3 residual, R5 parameter and sigma differences are all **0**.

| Case | Wall seconds | Result |
| --- | ---: | --- |
| R1 | 12.294 | Bit identical / zero-tolerance passes |
| R2 | 4.599 | Bit identical / zero-tolerance passes |
| R3 (cProfile) | 22.535 | Bit identical / zero-tolerance passes |
| R4 | 22.997 | Bit identical / zero-tolerance passes |
| R5 (4 ranks) | 134.026 | Bit identical / zero-tolerance passes |

Code lines from 354575f: library 23,029 → 22,899 (**−130**), tests 9,561
unchanged, scripts 953 unchanged; total 33,543 → 33,413 (**−130**).
Artifacts: `../refactor_work/elegance_phase4/`: 4_1_tests/ruff/mypy logs,
4_1_acceptance.log, per-case outputs/timings, R3–R5 comparator logs,
decoded_product_comparison.json, iteration_order_comparison.json,
tracked_line_counts.json, R3.prof and R3_top20.txt.

### R3 cProfile top 20 at 4.1

```text
Sun Nov  1 12:19:22 2026    ../refactor_work/elegance_phase4/R3.prof

         32330998 function calls (32323003 primitive calls) in 21.842 seconds

   Ordered by: cumulative time
   List reduced from 3040 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    421/1    0.041    0.000   22.400   22.400 {built-in method builtins.exec}
        7    0.025    0.004   18.128    2.590 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000   11.817   11.817 <string>:1(<module>)
      2/1    0.000    0.000   11.817   11.817 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    7.860    7.860 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    7.858    7.858 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    7.858    7.858 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:206(main)
        1    0.000    0.000    7.846    7.846 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    7.828    7.828 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.005    0.005    7.788    7.788 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:115(rows)
      494    0.009    0.000    7.703    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.025    0.000    7.628    0.015 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:353(solve)
        1    0.000    0.000    4.693    4.693 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.018    0.000    3.675    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:219(_station_state_at_utc)
     2416    0.023    0.000    3.560    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:62(displacement_itrf_m)
     1925    0.002    0.000    3.521    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:274(_station_state_from_tdb)
        1    0.000    0.000    3.257    3.257 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.142    0.142    3.253    3.253 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:208(parse_mini_file)
        2    0.112    0.056    2.598    1.299 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:379(_read_iers)
     2416    0.016    0.000    2.576    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/solid_earth_tide.py:52(displacement_itrf_m)
```

## Elegance Phase 4 — stopped at 4.2 under §2.10 (2026-11-01)

Accepted implementation commit: **ab172fa** (4.1), +153/−213 including this
log, net **−60**; code net **−130**. R1–R5 results, wall times, line counts
and R3 cProfile top-20 are recorded immediately above. Phase 4 is incomplete.
4.2, 4.4, 4.5, 4.6 and the separately approved numerical item 4.3 have not been
implemented. The 2.4 record was read: explicit observation dependency injection
is already established and must be retained when 4.6 resumes.

### Pending plan revision: per-station geodetic cache

The prescribed 4.2 cache (one latitude/height per station in the resolver, with
no invalidation) cannot preserve the current calculation. `StationRecord`
propagates reference XYZ with its nonzero secular velocity to each NP's
transmit UTC, and `LightTimeRequest.from_station` converts that propagated XYZ
to geodetic latitude and height. R3/R5 use nonzero station velocities. The
station catalog also changes during station-parameter adjustment.

A read-only probe of all **494 frozen R3 observations**, caching the geodetic
coordinates at each station's first selected observation, gives maximum
latitude change **5.229666699690938e-10 rad** and height change
**0.0004200739786028862 m**. For APOLLO, 2019-01-17T06:14:55 to
2019-03-28T12:46:15, cached/current heights are 2786.72103357967 /
2786.7214536536485 m. Holding that observation's meteorology and a 30-degree
leg elevation fixed, the production Mendes–Pavlis model's path delay changes
by **4.966800304373464e-10 m** when reusing the cached latitude/height.
This probe is evidence that the proposed cache changes physical inputs and
path delays; it is not a measured R3 residual difference or an R5 rerun with
caching. It conflicts with the required bit-identical behavior, even though
the delay difference is small. No cache or numerical change was committed.

Evidence: `../refactor_work/elegance_phase4/probe_station_cache.py` and
`station_cache_infeasibility.json`. Resolving this requires revising the cache
target or allowing a numerical change; no epoch-dependent cache keys,
invalidation, fallback, or tolerance relaxation were introduced. Per §2.10,
stop the whole sequence here because the prescribed change is infeasible
under its numerical requirement. Do not classify this as a dead-code target
that may simply be skipped. 4.3 remains authorized but pending the preceding
items; its numerical acceptance results are not yet available.

## Elegance Phase 4.2 — shared measurement inputs and diagnostic declarations

User-approved plan revision d863074 removes the per-station geodetic cache;
this resolves the bb9c312 blocker. Geodetic coordinates remain computed per NP.

- Both partial functions now receive the same uplink/downlink unit vectors;
  removed the 1e-30 norm clamps without changing division/summation order.
- One ObservationOptions object replaces repeated model keyword forwarding;
  callers and tests migrated, no compatibility alias.
- FULL_DIAGNOSTIC_FIELDS declares (name, unit, getter); the same evaluated row
  remains the input for residual output/report consumers. ASCII result schema
  reads/writes declared units. Diagnostic names/order/arithmetic stay unchanged.
- Full diagnostic row, field order and all four design vectors also compared
  bit-for-bit with ab172fa in a direct audit (full_diagnostics_4_2_comparison.json).
- 544 tests pass (7.69 s), ruff clean, mypy 0 errors (211 files).
- R1/R2 orbit and integrated rotation bit identical to Phase 0; diagnostics bit
  identical to Phase 3.1 native reference. R3/R4/R5 zero-tolerance pass. Decoded
  R5 parameters/sigma/normals/covariance exact, joint 16 iterations converged.
  R3 residual / R5 parameter / sigma maximum differences: **0 / 0 / 0**.

Wall seconds: R1 12.856, R2 4.579, R3 22.466, R4 23.099, R5 133.512.

Code diff +220/−194, net **+26** (tests net 0). The small increase is
from explicit diagnostic getter context/unit metadata and reader/writer unit
wiring; it replaces the hand-written row dict and repeated keyword lists.
Artifacts: ../refactor_work/elegance_phase4/4_2/ plus 4_2_tests/ruff/mypy.log
and audit_4_2.py in the parent. This run initially reused the root phase4 runner
and its stale "4.1" label; its outputs/profile/timings/comparisons are now copied
into 4_2/. Future item runners use their own directory; the root R1–R5 outputs
currently contain the accepted 4.2 products, superseding the earlier 4.1 run.

## Elegance Phase 4.4 — one range-bias table

- RangeBiasTable replaces Request/Correction/Lookup/LookupStatus/ABC/ZeroModel/
  TableModel and the separate AdditiveRangeBiasTable wrapper. Empty table is the
  zero model; lookup(station, date) returns (cm, components). Component ordering
  and the correction arithmetic are unchanged. Coverage/source/unit diagnostic
  data is constructed only for FULL output.
- All three config types instantiate RangeBiasTable through its schema/factory;
  no registry API change, compatibility class, alias or fallback. RangeBiasComponent
  is the interval payload retained for 4.5.
- 544 tests pass (7.64 s), ruff clean, mypy 0 errors (210 files).
- R1/R2 phase0 orbit/integrated state and 3.1-native diagnostics exact. R3/R4/R5
  zero-tolerance pass; decoded R5 parameters, sigma, normals and covariance exact,
  16 joint iterations converged. Maximum R3 residual / R5 parameter / sigma
  differences: **0 / 0 / 0**. Direct FULL audit covers empty table, overlapping
  components, explicit zero, outside coverage and unknown station: every scalar
  and design vector bit identical to e366896.
- Code diff +140/−295, net **−155** (library −148, tests −7).
- Artifacts: ../refactor_work/elegance_phase4/4_4/, 4_4_tests/ruff/mypy.log,
  audit_4_4.py and full_diagnostics_4_4_comparison.json.

Wall seconds: R1 12.086, R2 4.458, R3 22.280, R4 23.220, R5 134.077.

## Elegance Phase 4.5 — shared station/date intervals

- StationInterval supplies canonical date parsing, station matching and [start,
  end) membership, with open end supported. Range-bias components and variance
  definitions contain it; station-bias parametrizations use it directly.
- Removed StationBiasInterval, _iso_date and _parse_date; _date_text was already
  removed in Phase 2. Builtin slash-date conversion delegates to the same parser.
  Callers/exports/tests migrated with no aliases. Interval keys, component order,
  adjustment report date strings and parameter names remain unchanged.
- 544 tests pass (7.56 s), ruff clean, mypy 0 errors (211 files). R1/R2 orbit,
  integrated states and native diagnostics exact; R3/R4/R5 zero-tolerance pass.
  Decoded R5 products exact and joint 16 iterations converged; maximum R3 residual,
  R5 parameter and sigma differences **0 / 0 / 0**.
- Code +100/−132, net **−32** (library −35, tests +3); the new 35-line module
  replaces more interval/parser code than it adds, as required by §2.12.
- Artifacts: ../refactor_work/elegance_phase4/4_5/, 4_5_tests/ruff/mypy.log.

Wall seconds: R1 12.157, R2 4.587, R3 22.424, R4 23.157, R5 153.800.

## Elegance Phase 4.6 — optional components and dataset ownership

- Removed gravitational/tropospheric ABCs and Zero delay/displacement classes;
  disabled declarations resolve to absence and assembly omits them from component
  tuples. Native delay classes are concrete; the single reflector-displacement
  Protocol was removed, while StationDisplacement remains for test fakes.
- Kept Phase 2.4's explicit ephemeris/frame/EOP injection. _ObservationDependencies
  and factory_context were already removed by 91bdfcf; no new injection mechanism.
- NptDataset shares metadata instead of deepcopy; no-time-filter returns self.
  Direct audit checks both ownership requirements and exact FULL row/partial
  comparisons against 1cba685, with absent and active troposphere.
- 544 tests pass (7.61 s), ruff clean, mypy 0 errors (211 files). R1/R2 phase0
  orbit/integrated state and native diagnostics exact; R3/R4/R5 zero-tolerance
  pass, decoded R5 products exact, joint 16 iterations converged. Maximum R3
  residual / R5 parameter / sigma differences **0 / 0 / 0**.
- Code +53/−171, net **−118** (library −104, tests −14).
- Artifacts: ../refactor_work/elegance_phase4/4_6/, 4_6_tests/ruff/mypy.log,
  audit_4_6.py and full_diagnostics_4_6_comparison.json.

Wall seconds: R1 12.263, R2 4.466, R3 22.553, R4 23.227, R5 153.972.

## Elegance Phase 4.3 — precise receive UTC for station partials

User-authorized numerical change, performed after all bit-identical Phase 4
items were accepted. LightTimeSolution now exposes receive_epoch_utc from the
last evaluated receive-station event; the station partial uses that exact UTC.
This UTC was already computed internally, but had not been a public solution
field. No extra time conversion, fixed-point evaluation or tolerance change.

Section 3.3 gate against frozen Phase 0 **passes**:

- R3 maximum one-way residual difference: **0 m** (limit 1e-9 m).
- R5 maximum parameter difference: **2.9531932455029164e-14** in its parameter
  unit, **4.217063882453421e-14 sigma** (limit 1e-6 sigma), for
  moon:libration.rho_l_27.6d.cos::.
- R5 maximum sigma absolute difference: **4.601874437071274e-14**; maximum
  relative difference **1.8103056970822178e-13** (limit 1e-9), at MATERA:position.z::.
- R5 joint iterations **16 → 16**, sigma/weight iterations **160 → 160**,
  converged. R1/R2 orbit, R1 integrated rotation against Phase 0, and both native
  diagnostic tables against 3.1 remain bit identical; R4 zero-tolerance passes.
- 544 tests pass (7.55 s), ruff clean, mypy 0 errors (211 files).
- Code +3/−1, net **+2**: two lines expose the existing UTC event as a field and
  constructor argument; the partial replaces one expression. No Cython edits.

Accepted new R3/R5 outputs are frozen in
`../refactor_work/elegance_phase4/numeric_reference/{R3,R5}/`; SHA256.json covers
all 30 files. Later phases compare R3/R5 to this reference bit-for-bit. Other
references stay Phase 0 (R1/R2 diagnostics stay the 3.1 native reference).
Evidence: 4_3/numeric_comparison.json, numeric_compare.log, run logs/products,
and integrated_rotation_comparison.json. The initial comparison script could
not JSON-serialize ParameterName; converting report labels to strings fixed
that audit-only error. The same simulation outputs were then accepted without
rerunning or changing data. R5's zero-tolerance log shows the allowed numerical
changes; it is not the acceptance criterion for this numerical item.

Wall seconds: R1 12.099, R2 4.441, R3 22.508, R4 23.409, R5 150.650.

## Elegance Phase 4 — final acceptance (2026-11-01)

All items 4.1–4.6 are accepted. The user-approved 4.3 numerical change was
performed last and separately, after 4.2/4.4/4.5/4.6 had passed bit-identical
regressions. The geodetic-cache requirement was withdrawn, without replacing
it with any other cache. No physics selection or convergence rule changed.
Phase 5 has not started.

### Commits and net lines

| Item | Commit | Git additions/deletions including progress | Git net | Code net |
| --- | --- | ---: | ---: | ---: |
| 4.1 unified light-time loop/legs | ab172fa | +153/−213 | −60 | −130 |
| Plan: withdraw geodetic cache | d863074 | +1/−1 | 0 | 0 |
| 4.2 vectors/options/diagnostic fields | e366896 | +251/−194 | +57 | +26 |
| 4.4 range-bias table | 51bedfb | +163/−295 | −132 | −155 |
| 4.5 station intervals | 1cba685 | +119/−132 | −13 | −32 |
| 4.6 optional components/metadata | 16c710f | +74/−171 | −97 | −118 |
| 4.3 precise receive UTC | 5dc0347 | +39/−1 | +38 | +2 |

The interim blocker record bb9c312 is superseded by the approved plan revision.
Git counts above use `git show --stat`; rewrite detection can change gross
additions/deletions without changing net lines.

Complete tracked .py/.pyx/.pxi/.pyi newline counts from Phase 3 accepted 354575f
to final implementation 5dc0347 (counts include removed and newly added files):

| Scope | Before | After | Net |
| --- | ---: | ---: | ---: |
| Library | 23,029 | 22,640 | **−389** |
| Tests | 9,561 | 9,543 | **−18** |
| Scripts | 953 | 953 | 0 |
| All code | 33,543 | 33,136 | **−407** |

The reduction is smaller than the approximate −700 estimate: the assembly
injection portion of 4.6 was already done in Phase 2.4, geodetic caching was
withdrawn, and 4.2's explicit diagnostic context/unit metadata adds 26 net code
lines. The 4.3 increase exposes the receive UTC already used by the solver.

### Final R1–R5

The complete suite was rerun after the numerical item. Frozen Phase 0 configs
were unchanged; R5 used four MPI ranks. Each earlier nonnumerical item passed
all five cases with zero tolerance, including decoded R5 normal equations,
parameter/sigma vectors and covariance. Final acceptance after 4.3:

| Case | Wall seconds | Result |
| --- | ---: | --- |
| R1 | 12.099 | Phase 0 orbit and integrated rotation exact; 3.1 native diagnostics exact |
| R2 | 4.441 | Phase 0 orbit exact; 3.1 native diagnostics exact |
| R3 (cProfile) | 22.508 | Residual max difference 0 m; zero-tolerance table comparison passes |
| R4 | 23.409 | Zero-tolerance comparison passes |
| R5 (4 ranks) | 150.650 | Converged; numerical gate passes; 16 joint / 160 sigma-weight iterations |

R1 integrated comparison directly includes quaternions, mantle/core angular
velocity and quaternion norm error against Phase 0. The full rotation table
(including diagnostics) and both force diagnostic tables match the accepted
3.1 native reference. No comparison to Phase 0 Python diagnostics is used.
R5L and R6 are not scheduled for Phase 4.

4.3 maximum changes against Phase 0: R3 residual **0 m**; R5 parameter
**2.9531932455029164e-14** in its parameter unit, normalized maximum
**4.217063882453421e-14 sigma**; sigma absolute maximum
**4.601874437071274e-14**, relative maximum **1.8103056970822178e-13**.
All meet §3.3, with joint and sigma/weight iteration counts unchanged.

Final §3.1 checks: 544 tests pass; ruff clean; mypy 0 errors in 211 source files.
No .pyx/.pxi changes in Phase 4, so no OpenMP rebuild was required.

### References and artifacts

- Final run: `../refactor_work/elegance_phase4/4_3/`; preceding bit-identical
  acceptance: `4_2/`, `4_4/`, `4_5/`, `4_6/` under the same artifact root.
- New future-phase R3/R5 reference:
  `../refactor_work/elegance_phase4/numeric_reference/{R3,R5}/`.
  All 30 copied product files are SHA-256 verified against SHA256.json;
  numeric_reference_metadata.json records source commit 5dc0347 and gate results.
  Later phases compare these R3/R5 products bit-for-bit. R1/R2 references remain
  Phase 0 orbit/integrated state and Phase 3.1 native diagnostics.
- Evidence: 4_3/numeric_comparison.json and numeric_compare.log, per-item
  acceptance/zero-tolerance logs, decoded_product_comparison.json, direct FULL
  audits, integrated_rotation_comparison.json, tracked_line_counts_final.json,
  final_tests.log, final_ruff.log and final_mypy.log. Root-level R1–R5 outputs
  were superseded by 4.2; use the numbered item directories for accepted results.
- Final R3 profile: 4_3/R3.prof and 4_3/R3_top20.txt, copied below.

### Final R3 cProfile top 20 (cumulative)

```text
Sun Nov  1 13:04:06 2026    ../refactor_work/elegance_phase4/4_3/R3.prof

         32351334 function calls (32343359 primitive calls) in 21.837 seconds

   Ordered by: cumulative time
   List reduced from 3024 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    418/1    0.040    0.000   22.397   22.397 {built-in method builtins.exec}
        7    0.024    0.003   18.141    2.592 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000   11.859   11.859 <string>:1(<module>)
      2/1    0.000    0.000   11.858   11.858 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    7.902    7.902 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    7.901    7.901 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    7.901    7.901 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:206(main)
        1    0.000    0.000    7.888    7.888 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    7.882    7.882 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.004    0.004    7.843    7.843 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:71(rows)
      494    0.008    0.000    7.760    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.027    0.000    7.691    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:351(solve)
        1    0.000    0.000    4.684    4.684 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.018    0.000    3.684    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:218(_station_state_at_utc)
     2416    0.022    0.000    3.568    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:35(displacement_itrf_m)
     1925    0.002    0.000    3.527    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:273(_station_state_from_tdb)
        1    0.000    0.000    3.261    3.261 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.141    0.141    3.257    3.257 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:208(parse_mini_file)
     2416    0.016    0.000    2.588    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/solid_earth_tide.py:52(displacement_itrf_m)
        2    0.112    0.056    2.574    1.287 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:379(_read_iers)
```

## Elegance Phase 5 prerequisite — cb3c29d R5L reference (2026-11-01)

Before any implementation change, ran the unchanged cb3c29d checkout via Slurm
job **44989840**, 16 MPI ranks, node26, 2012–2020, all parameters. Job completed
with exit 0; wall time **299.701 s**. Outputs are stored in
`../refactor_work/elegance_phase4/numeric_reference/R5L/`.

Against Phase 0 R5L:

- Final rank **44/44 → 44/44**, each joint-iteration rank remains 44.
- Joint iterations **8 → 8**, sigma/weight iterations **80 → 80**; intermediate
  reflector/bias/joint iteration counts remain **8 / 3 / 8**.
- Converged and formalProductEligible true.
- Maximum parameter absolute difference **8.881784197001252e-15**, normalized
  maximum **3.0143416204137236e-14 sigma** (moon:libration.rho_l_27.6d.sin::).
- Maximum sigma absolute difference **6.467049118441537e-15**, relative maximum
  **1.2319480928582315e-13** (WETTZELL:rangeBias:interval:2018-01-01/present).
- Section 3.3 limits pass. No code changed to obtain this reference.

Evidence: ../refactor_work/elegance_phase5/baseline_cb3c29d/ contains submission,
Slurm script/accounting/log, comparison.json/log and R5L_SHA256.json. R5L run
log/timing JSON live alongside the new numeric reference. Phase 5 uses Phase 4
numeric references for R3/R5/R5L, Phase 0 for R4 and R1/R2 orbit/integrated state,
and Phase 3.1 native diagnostic references for R1/R2 diagnostics.

## Elegance Phase 5 — stopped at 5.1 regression under §2.10 (2026-11-01)

Only the prerequisite reference record is accepted/committed: **78a1d19**
(documentation +25 lines; implementation unchanged from cb3c29d). R5L reference
Slurm job 44989840 passed with unchanged rank/iteration counts and §3.3 parameter/
sigma differences, as recorded above. Phase 5 implementation is **incomplete**.

### Rejected 5.1 attempt and regression failure

- Implemented an array-only EquationBatch, fixed partial columns, catalog index
  arrays and receive epochs; removed ObservationEquation and all partial dicts.
  Tests/callers were migrated without a compatibility alias. Single-row slices
  were used by existing row algorithms, pending the block methods of 5.2.
- User clarified MPI scope: necessary batch payload migration is allowed;
  task splitting, source ordering and stable observation-id sort remain unchanged.
  Worker payloads became batches; concatenate plus stable argsort preserved order.
  No Phase 6 scheduling/merge redesign was attempted.
- Attempt checks: **544 tests passed in 7.66 s**, ruff clean, mypy 0 errors in
  211 files. R1/R2 orbit/integrated state exact to Phase 0, diagnostics exact to
  3.1 native reference; R3 vs Phase 4 numeric reference and R4 vs Phase 0 pass
  zero-tolerance comparison.
- Before editing, a 4-rank R5 instrumented run captured all **29 linearizations**
  from accepted code and passed output comparison to the Phase 4 numeric reference.
  Rejected 5.1 captured all 29 again: **A, l, sigma, P, active masks and x0 are
  bit identical in all 58 array files**. No numerical tolerance was invoked.
- However the 4-rank R5 acceptance run exits **1** after 16 joint iterations,
  during `_write_residuals` → `write_observation_results` → `_field_type`:
  **TypeError: Table fields must have one consistent scalar type.** The equation
  flag columns yield numpy.bool_ scalars in report observation rows, while the
  existing table writer recognizes Python bool. A direct probe confirms Python
  bool is accepted and numpy.bool_ fails. Numeric solve equality does not satisfy
  acceptance when final products cannot be written.
- The test suite did not catch this output boundary type change. Both actual
  R5 runs (ordinary and instrumented) fail at that final write. No R5 output
  equality or Phase 5 acceptance is claimed. No 5.1 R5L was run.

Per §2.10, stopped immediately on the regression, saved the rejected patch and
restored **all implementation/test edits**. No writer exception, type fallback,
clamp, compatibility layer or post-failure implementation fix was introduced.
5.1 is not committed; 5.2–5.5 have not been implemented. Final code net lines
relative to cb3c29d are **0**. The rejected proposal was +387/−298, net **+89**
(library +80, tests +9): column storage/single-row construction and catalog
concatenation added infrastructure; scalar consumers had not yet been eliminated
by the planned 5.2 step. The temporary increase is recorded, not accepted.

### Checks prepared for 5.2

On all **9,626 R5L epochs**, erfa.fal03/falp03/faf03/fad03/faom03 called with an
array are each bit-identical to scalar calls (maximum differences 0). This is
only a preliminary probe; no 5.2 code was applied or accepted.

### Attempt wall times

| Case | Wall seconds | Result |
| --- | ---: | --- |
| R1 | 12.091 | Bit identical / zero tolerance passes |
| R2 | 4.456 | Bit identical / zero tolerance passes |
| R3 (cProfile) | 22.526 | Bit identical / zero tolerance passes |
| R4 | 23.333 | Bit identical / zero tolerance passes |
| R5 (4 ranks) | 169.435 | FAILED at final residual write |
| R5L prerequisite (Slurm, 16 ranks, cb3c29d) | 299.701 | Reference accepted; rank 44/44, iterations 8/80 |

### Evidence

Artifacts under ../refactor_work/elegance_phase5/:

- baseline_cb3c29d/: submission/accounting, comparison and reference manifest.
- rejected_5_1_equation_batch.patch, rejected_5_1_diffstat.txt,
  rejected_5_1_line_counts.json; patch applies to 78a1d19.
- 5_1/: attempted run outputs/logs/timings, R3/R4 comparisons, R3.prof/top20.
- dense_reference/, 5_1_dense/, their logs and 5_1_dense_comparison.json.
- 5_1_failure_evidence.json and 5_1_tests/ruff/mypy.log.
- probe_erfa_array.py and erfa_array_probe.json.

The accepted R5L reference remains at
../refactor_work/elegance_phase4/numeric_reference/R5L/ and is unaffected by
restoring the rejected source edits. The earlier R3/R5 reference manifest remains
valid; the added R5L subtree has a separate manifest in baseline_cb3c29d/.

### R3 cProfile top 20 from rejected 5.1 attempt

```text
Sun Nov  1 13:46:48 2026    ../refactor_work/elegance_phase5/5_1/R3.prof

         32366868 function calls (32358881 primitive calls) in 21.857 seconds

   Ordered by: cumulative time
   List reduced from 3026 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    418/1    0.039    0.000   22.416   22.416 {built-in method builtins.exec}
        7    0.023    0.003   18.172    2.596 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000   11.889   11.889 <string>:1(<module>)
      2/1    0.000    0.000   11.889   11.889 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    7.943    7.943 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    7.941    7.941 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    7.941    7.941 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:206(main)
        1    0.000    0.000    7.930    7.930 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    7.923    7.923 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.006    0.006    7.885    7.885 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:71(rows)
      494    0.009    0.000    7.794    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:131(evaluate)
      494    0.028    0.000    7.711    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:351(solve)
        1    0.000    0.000    4.676    4.676 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.018    0.000    3.702    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:218(_station_state_at_utc)
     2416    0.022    0.000    3.588    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:35(displacement_itrf_m)
     1925    0.002    0.000    3.538    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:273(_station_state_from_tdb)
        1    0.000    0.000    3.244    3.244 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.139    0.139    3.241    3.241 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:208(parse_mini_file)
     2416    0.016    0.000    2.592    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/solid_earth_tide.py:52(displacement_itrf_m)
        2    0.115    0.057    2.567    1.283 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:379(_read_iers)
```

Restored accepted code checks: 544 passed in 7.71 s, ruff clean, mypy 0 errors
(211 files); rejected patch passes `git apply --check` against restored source.

## Phase 5.1 retry — accepted (2026-11-01)

Plan revision committed first as `5bbe8e9`. Reapplied
`../refactor_work/elegance_phase5/rejected_5_1_equation_batch.patch`.
The previous failure was producer-side `numpy.bool_` in report rows: the
table writer correctly requires the original native Python field types.
Report rows now zip whole columns converted with `.tolist()`; the standard
observation sigma uses `.item()`. No writer compatibility branch was added.
The new R5-shaped `test_r5_batch_report_rows_write_native_bool_columns`
constructs real bias/report rows and writes the FULL residual artifact through
`write_observation_results`, covering both true/false convergence/clamping.

R1/R2 orbit and integrated state match phase 0 bit for bit; diagnostics match
the Phase 3 diagnostic reference bit for bit. R3/R5 compare against Phase 4
numeric_reference with zero tolerance, R4 against phase 0: all pass. R5
parameters, sigma, N/W/x0/covariance and rank/iterations agree exactly.
Fresh capture `5_1_retry_dense/` has 29 builds and 29 normals: all A/l/sigma,
P/active-mask/x0 and parameter names in 58 archives match `dense_reference/`
byte for byte (`5_1_retry_dense_comparison.json`). The repaired full R5 exits 0.

| Case | Wall seconds | Result |
| --- | ---: | --- |
| R1 | 12.086 | Bit identical |
| R2 | 4.432 | Bit identical |
| R3 (cProfile) | 22.443 | Zero tolerance |
| R4 | 23.496 | Zero tolerance |
| R5 (4 ranks) | 170.602 | Zero tolerance; full output written |

Checks: 545 passed (7.78 s), ruff clean, mypy 0 errors (211 files).
Artifacts: `../refactor_work/elegance_phase5/5_1_retry/`,
`5_1_retry_dense/`, comparison JSON and retry test/ruff/mypy logs.
MPI changes are limited to authorized batch payload migration: task partition,
source order and stable observation-id sorting stay the same. Parametrizations
still take singleton slices pending 5.2.

Implementation line counts (git diff --numstat, excluding this progress entry):
library +288/−202 (net +86); tests +146/−98 (net +48).
Temporary growth and R5 overhead come from the column container alongside
temporary singleton slicing. They are explicitly allowed for 5.1; the combined
5.1+5.2 net reduction and same-machine +3% performance gate remain mandatory.

## Phase 5.2 — accepted (2026-11-01)

Parametrizations now expose setup(batch), design(batch, out), x0(), update(dx),
and reduce(batch). Column offsets are computed during setup, and design writes
whole block slices. Removed row entries/design_row, repeated layout/name matching,
index-by-key maps, and unset-state guards. Position model state is injected through
registry kwargs, following the existing explicit injection mechanism. Requested
selection uniqueness belongs to ConfigSchema: FieldSpec.unique now enforces the
former constructor requirement at that boundary (the old schemas did not actually
check duplicates). All callers and fake blocks use the new contract.

Linear blocks share ordered reduction of physical contributions: bias intervals
remain in configured order and libration cosine/sine pairs include fixed terms.
This avoids changing the summation order of a BLAS dot product. Fundamental ERFA
arguments are computed on epoch arrays once and cached by EquationBatch. Scalar
trigonometric evaluation retains the original rounding. The new cache test checks
scalar ERFA equality and reuse; the earlier probe covered all 9,626 R5L epochs.
No numerical tolerance exception was needed. Batch slicing/iteration still have
production callers in observation screening, fixed-domain collection and component
reports, so they are retained under the explicit production-caller exception; no
parametrization block slices individual observations.

Fresh 5_2_dense capture: 29 linearizations and 29 normal-equation inputs; every
A/l/sigma/P/active-mask/x0/name array is byte identical to dense_reference (max 0).
R1/R2 orbit, integrated state and diagnostic references are bit identical.
R3/R4/R5 pass zero-tolerance comparisons against the required references; R5
parameters, sigma, normals and covariance have maximum difference 0, identical
rank and iterations. Parsed report objects are equal after excluding only
provenance/output paths and timing, as listed in the comparison JSON.

| Case | Wall seconds |
| --- | ---: |
| R1 | 12.170 |
| R2 | 4.438 |
| R3 (cProfile) | 22.484 |
| R4 | 23.512 |
| R5 (4 ranks) | 132.167 |

Performance gate, sequential alternating runs on rosefinch-login04, 4 ranks,
OMP/MKL/OpenBLAS threads all 1; cb3c29d is an unchanged detached checkout using
the same compiled extensions and frozen input config.

| Version | Trial 1 s | Trial 2 s | Trial 3 s | Median s |
| --- | ---: | ---: | ---: | ---: |
| cb3c29d | 150.539 | 149.672 | 150.223 | 150.223 |
| 5.2 | 132.957 | 131.742 | 132.375 | 132.375 |

Median change −11.8812%, passes ≤+3%. All six runs exit 0 and their decoded
parameters/sigma/normals/covariance and parsed reports agree with reference.

Checks: 546 tests passed (7.78 s), ruff clean, mypy no issues (211 files).
5.2 library +190/−383 = −193; tests +127/−81 = +46; code total −147.
Combined 5.1+5.2: library −107, tests +94, code total −13 (docs excluded).
Evidence: elegance_phase5/5_2/, 5_2_dense/, capture comparison JSON,
5_2_line_counts.json, 5_2_perf_records.json/5_2_perf_result.json, perf_* run
folders and 5_2_tests/ruff/mypy logs. No pyx/pxi change; no rebuild required.

## Phase 5.3 — accepted (2026-11-01)

Solver.solve returns explicit AdjustmentState; run delegates final report
assembly to adjustment_reporting.build_report(state, plan). A single
LlrAdjustmentResult serves intermediate and final steps, with intermediate
steps retaining the previous no-extra-forward-model/no-final-solve behavior.
Removed the six domain copies and _domain forwarding, consolidated ObsKey,
reused the already computed weight summaries, and grouped component observations
once in their original order. Parametrization state now emits endExclusive;
removed report_state and its rename patch. This planned processing-state key
change is checked semantically; scientific values are unchanged.

R1/R2 orbit/integrated states and Phase 3 diagnostics are bit identical; R3/R4
zero tolerance pass. R5 parameters/sigma/N/W/x0/covariance/residuals have max
difference 0, identical rank/iterations. Report YAML is compared as parsed
objects, excluding only provenance/output paths and timing; equal. The new
external comparator initially tried decoding a binary artifact while probing
its header; corrected only its probe to match compare_outputs' text handling
and reran all R5 comparisons successfully. Native decoded array comparison
independently confirms binary values byte for byte; no production correction
or numerical tolerance was involved. Original failure log remains available.

Wall seconds: R1 12.112; R2 4.445; R3 (profile) 22.330; R4 23.260;
R5 (4 ranks) 130.606.
Checks: 546 tests passed (7.71 s), ruff clean, mypy no issues (211 files).
Library +240/−251 = −11; tests +2/−2 = 0; code net −11.
Evidence: elegance_phase5/5_3/, 5_3_line_counts.json and test/ruff/mypy logs.

## Phase 5.4 — accepted (2026-11-01)

Observation report/residual fields now share a (name, unit, getter) declaration
and the diagnostic_values mechanism used by the Phase 4.2 full diagnostics.
Report rows come from typed native row values; residual projection derives its
names from the same declarations, removing the 26-entry hardcoded mapping.
Standard residual fields and full matched-parameter formatting remain unchanged.

R1/R2 orbit/integrated state and Phase 3 diagnostics are bit identical.
R3/R4/R5 zero-tolerance comparisons pass. R5 parameter/sigma/normal/covariance
and residual maximum differences are 0; rank and iterations unchanged.
Report YAML objects are equal (only provenance/timing excluded, key order free).
Wall seconds: R1 12.100, R2 4.426, R3 22.472, R4 23.285, R5 131.450.
Checks: 546 tests passed (7.68 s), ruff clean, mypy no issues (211 files).
Library +81/−71 = +10; tests unchanged. This expected reduction item instead
grows by 10 lines: the shared declaration now includes units and typed getter
context for all report fields, replacing implicit local-variable access; a
single diagnostic_values function serves physical and adjustment diagnostics.
This provides a common source of names/values rather than another writer map.
Evidence: elegance_phase5/5_4/, 5_4_line_counts.json and test/ruff/mypy logs.

## Phase 5.5 — accepted (2026-11-01)

Deleted report_schema.py, output-side processing/orbit validation and snapshot
schemas/validation (including private schemas used solely by snapshots). YAML
serialization now directly uses plain_data with the existing safe C dumper.
Readers retain their artifact header and mapping boundary; the restart reader
checks only keys actually consumed on restart. read_processing_report(path)
retains its signature. Formal-product gating remains in require_formal_product.
Added two reader tests for mismatched artifact headers. Output/regression_*
references and run_regression_suite.py are untouched; the known version-1
normalEquationMetadata compare issue is left for Phase 7.

Checks: 548 tests passed (7.59 s), ruff clean, mypy no issues (210 files).
R1–R5 and Slurm R5L pass the required comparisons. Scientific maximum differences
are 0; parsed report YAML objects are equal after only provenance/timing removal.
R5/R5L residual readers independently confirm native scalar types, bool/string
values, and every float's binary representation, including signed zero.
Slurm 44989861, node22, 16 ranks, COMPLETED 0:0; R5L 196.812 s, rank 44,
joint iterations 8, sigma/weight iterations 80, matching the reference.
The parallel final runner initially attempted decoded comparison before Slurm
had published R5L solution files; rerunning comparison after job completion
passed. This was an external harness ordering error, with no production error
or numerical difference. Original logs and corrected comparisons are retained.

Library +10/−744 = −734; tests +11/−0 = +11; code total −723.
Evidence: elegance_phase5/final/ (runs, Slurm script/submission/accounting,
parsed/decoded/residual comparisons), 5_5_candidate.patch, 5_5_line_counts.json
and 5_5_tests/ruff/mypy logs. Source manifest confirms no code change during
R1–R5/R5L execution. No pyx/pxi changes; existing OpenMP build is unchanged.

## Phase 5 — completed acceptance (2026-11-01)

Phase 5.1–5.5 are complete. Plan revision `5bbe8e9` was committed before
implementation. The cb3c29d R5L prerequisite and phase-0 comparison are recorded
in the earlier R5L baseline section (`78a1d19`); its reference remains unchanged.
All implementation items have independent commits:

| Item | Commit | Library net | Tests net | Code net |
| --- | --- | ---: | ---: | ---: |
| 5.1 | `f216e63` | +86 | +48 | +134 |
| 5.2 | `97a4f76` | -193 | +46 | -147 |
| 5.3 | `513b29d` | -11 | +0 | -11 |
| 5.4 | `bae0fad` | +10 | +0 | +10 |
| 5.5 | `c3bc4b9` | -734 | +11 | -723 |

Line counts use tracked py/pyx/pxi files and newline counts (wc -l equivalent),
compared to accepted Phase 4 cb3c29d:

| Source | Phase 4 | Phase 5 | Net |
| --- | ---: | ---: | ---: |
| Library | 22,557 | 21,715 | −842 |
| Tests | 9,543 | 9,648 | +105 |
| Code total | 32,100 | 31,363 | −737 |

Docs/progress are excluded from code totals. 5.1+5.2 combined library −107,
code including tests −13: the required combined net reduction passes. The
5.2 same-machine three-run R5 performance gate passes at −11.8812%: 132.375 s
median against cb3c29d 150.223 s. The final single R5 run is 130.057 s.

### Final regressions and wall clocks

| Case | Ranks / node | Wall seconds | Reference and result |
| --- | --- | ---: | --- |
| R1 | 1 / login04 | 12.106 | phase 0 orbit/integrated state, Phase 3 diagnostics: bit identical |
| R2 | 1 / login04 | 4.443 | phase 0 orbit, Phase 3 diagnostics: bit identical |
| R3 | 1 / login04 | 22.592 | Phase 4 numeric reference: all native residual fields and float bits identical |
| R4 | 1 / login04 | 23.139 | phase 0: zero tolerance pass |
| R5 | 4 / login04 | 130.057 | Phase 4 numeric reference: parameter/sigma/residual/normal/covariance bits identical |
| R5L | 16 / Slurm node22 | 196.812 | Phase 4 numeric reference: parameter/sigma/residual/normal/covariance bits identical |
| R6 | — | — | Not scheduled in Phase 5 (§3.2) |

R1 includes quaternion, mantle/core omega and norm-error output; these are bit
identical. R3/R5/R5L maximum residual difference is **0 m**. R5/R5L maximum
parameter and sigma differences, including normalized/relative differences,
are **0**. R5 rank 26 and iterations 16/160; R5L rank 44 and iterations 8/80,
all unchanged. The R5L per-step outer counts remain reflector/bias/joint 8/3/8.
Slurm job **44989861**, 16 ranks, COMPLETED 0:0; accounting in final/.
R5/R5L parsed report YAML objects are equal after only provenance/output paths
and timing exclusions (explicit key lists are saved). YAML key order is free.
The processing-state endExclusive rename is the change explicitly required by
5.3; other state content is equivalent. No numerical gate exception was used.

Final §3.1 checks: **548 passed in 7.59 s**, ruff clean, mypy no issues in
210 source files. No pyx/pxi changes occurred in Phase 5; the existing OpenMP
extensions were reused. Output/regression_* and run_regression_suite.py have
no changes; the known version-1 metadata issue remains deferred to Phase 7.

### Remaining planned deletion target

EquationBatch.__getitem__/__iter__ still have production callers in screening,
fixed-domain equation collection and component statistics. They are retained
under the explicit 5.2 condition, and recorded for later cleanup; no compatibility
API was added. Parametrizations do not take singleton batch slices. Other Phase 5
targets were completed; MPI changes stop at necessary batch payload migration.

### Artifacts

All artifacts are under ../refactor_work/elegance_phase5/. final/ contains
acceptance_result.json (accepted=true), source/reference integrity manifests,
commit/whole-phase line counts, native decoded and residual-bit comparisons,
parsed YAML comparisons, all six run logs/results, Slurm script/submission/
accounting, and R3.prof/R3_top20.txt. 5_1_retry_dense and 5_2_dense contain the
29-by-29 captures; perf_* folders contain six same-machine R5 runs. The two
external harness errors were resolved by corrected header probing and waiting
for Slurm publication; original errors are retained beside successful comparisons.
No production numerical regression occurred.

### R3 cProfile top 20

```text
Sun Nov  1 15:12:11 2026    ../refactor_work/elegance_phase5/final/R3.prof

         32357088 function calls (32349258 primitive calls) in 21.920 seconds

   Ordered by: cumulative time
   List reduced from 2994 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    399/1    0.037    0.000   22.476   22.476 {built-in method builtins.exec}
        7    0.024    0.003   18.278    2.611 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000   11.976   11.976 <string>:1(<module>)
      2/1    0.000    0.000   11.976   11.976 <frozen runpy>:206(run_module)
      2/1    0.000    0.000    8.070    8.070 <frozen runpy>:65(_run_code)
      2/1    0.000    0.000    8.068    8.068 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    8.068    8.068 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:206(main)
        1    0.000    0.000    8.056    8.056 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    8.049    8.049 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.006    0.006    8.011    8.011 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/processor.py:71(rows)
      494    0.009    0.000    7.917    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/measurement.py:132(evaluate)
      494    0.027    0.000    7.832    0.016 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:351(solve)
        1    0.000    0.000    4.661    4.661 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
     2419    0.018    0.000    3.751    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:218(_station_state_at_utc)
     2416    0.024    0.000    3.632    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/base.py:35(displacement_itrf_m)
     1925    0.002    0.000    3.599    0.002 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/observation/light_time.py:273(_station_state_from_tdb)
        1    0.000    0.000    3.256    3.256 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.144    0.144    3.252    3.252 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:208(parse_mini_file)
     2416    0.016    0.000    2.631    0.001 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/displacement/solid_earth_tide.py:52(displacement_itrf_m)
        2    0.113    0.057    2.573    1.287 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:379(_read_iers)
```

## Phase 6 — completed acceptance (2026-11-02)

Starting point: accepted Phase 5 `e3bfba3`. No Phase 7 work or output-format changes
were made. All Phase 6 targets are completed:

- One scheduler: map_tasks is list(stream_tasks); ordered yields preserve
  original payload order even when worker completion order differs. Each worker
  has at most two tasks in flight. Early close and task failures drain pending
  work before returning. Progress uses tqdm, with batches as the explicit unit.
- Only TASK/RESULT/STOP tags remain. broadcast_spec(kind, spec) returns an integer
  ID, deduplicates a spec, and immediately builds worker processors/predictors/
  epoch ephemerides. Initialization elapsed times and remote errors are gathered
  collectively, so every worker finishes the collective before an error is raised.
- WorkerState holds explicit specs, contexts, processors, predictors, ephemerides
  and the shared immutable class cache. Resource shutdown iterates contexts
  directly; removed tuple-key cache scans and lazy prediction fallback layers.
- Merged observation/epoch spec preparation and observation equations/rows APIs.
  All production callers and tests use broadcast_spec and mpi_observations(rows=).
  Observation task/source order, stable observation-id sorting and EquationBatch
  concatenation remain unchanged. Removed old aliases and protocol defense checks.
- Rewrote protocol-oriented MPI tests around public behavior. The real three-rank
  public API test exercises repeated/distinct specs, eager initialization, remote
  initialization/task failures, task-order results, stream early close/drain,
  scheduling after errors and resource closure with shared immutable cache.
  Existing catalog/resource behavior and prediction-order tests remain covered.

### Checks and source lines

Final checks: **538 tests passed in 8.27 s**, ruff clean, mypy no issues in
210 source files. MPI tests were executed, not skipped. The count is 10 lower
than Phase 5 because test_mpi_spec_broadcast.py's 14 tests became 4: three catalog/
resource behavior tests and one real multi-process test replacing 11 internal
protocol tests. No pyx/pxi edits occurred; no OpenMP rebuild was required.

Line counts compare tracked py/pyx/pxi files with e3bfba3 (wc -l equivalent):

| Source | Phase 5 | Phase 6 | Net |
| --- | ---: | ---: | ---: |
| Library | 21,715 | 21,373 | −342 |
| Tests | 9,648 | 9,379 | −269 |
| Code total | 31,363 | 30,752 | −611 |

mpi.py alone is 733 → 389 lines (−344). Docs are excluded from code totals.
No dead-code deletion target needed to be skipped. Phase 5's retained
EquationBatch slicing/iteration consumers remain outside this phase's scope.

### Regression results and wall times

All runs use the unchanged frozen Phase 0 configs. Thread counts are
OMP/MKL/OpenBLAS=1. R1 uses MPI epoch prefetch; R3 additionally exercises MPI
observation rows. R5L/R6 ran through Slurm on node22, concurrently for part of
their runs, so these wall times are not a controlled speed benchmark.

| Case | Ranks | Wall seconds | Result |
| --- | ---: | ---: | --- |
| R1 | 4 | 11.781 | Phase 0 orbit/integrated state and Phase 3 diagnostics bit identical |
| R2 | 1 | 4.577 | Phase 0 orbit and Phase 3 diagnostics bit identical |
| R3 | 4 | 19.358 | Phase 4 numeric reference: zero tolerance and native float bits identical |
| R4 | 1 | 23.646 | Phase 0: zero tolerance |
| R5 | 4 | 133.743 | Phase 4 numeric reference: parameter/sigma/residual/normal/covariance bit identical |
| R5L | 16 | 196.512 | Phase 4 numeric reference: parameter/sigma/residual/normal/covariance bit identical |
| R6 | 9 | 107.652 | Phase 0 orbit/integrated state and accepted native diagnostics bit identical |

Slurm **44991349** (R5L, 16 ranks) and **44991350** (R6, 9 ranks) both completed
with exit 0:0. R6 uses the same rank/thread layout as
jobs/rotation_de440_environment_10a_mpi.sbatch, with the frozen R6 config and
Phase 6 output location. It has 3,654 orbit/rotation samples and 3,653
acceleration diagnostic samples. Every quaternion, mantle/core omega and
quaternion norm error agrees with Phase 0; maximum Moon position difference
is **0 mm**. Full rotation/force diagnostics match elegance_phase3_finish/R6
byte for byte. The accepted Phase 3 Python-to-Cython diagnostic differences
against Phase 0 remain unchanged; no new diagnostic tolerance was introduced.

R5 rank **26**, outer/sigma-weight iterations **16/160**; R5L rank **44**,
iterations **8/80**, all unchanged from the Phase 4 numeric reference. Maximum
parameter and sigma differences (including normalized/relative differences)
are **0**. R3/R5/R5L maximum residual differences are **0 m**. Independent
reader comparisons verify native scalar types and every residual float's bits
(including signed zero), not just numeric text equality. Report YAML objects
are equal after the existing provenance/path/timing exclusions; key order
is ignored. Normals, lPl/count, covariance, parameter ordering and convergence
all agree exactly. No numerical exception or relaxed gate was used.

Additional R4 MPI test (4 ranks, 15.289 s): all 4,321 prediction rows and the
one window, including every native scalar/float bit, match Phase 0. Its only
metadata change is the expected executionMode serial → mpi. The initial
directory comparator flagged that provenance value; the required serial R4
was then rerun and passed without exclusions. The extra MPI output and
metadata comparison remain as R4_MPI/. Production code and baseline files
were not changed to accommodate this execution-mode difference.

### Profiling and artifacts

Artifacts are in ../refactor_work/elegance_phase6/. final/ contains accepted=true
in acceptance_result.json, all run logs/results, Slurm scripts/submissions/
accounting, decoded/native residual/R6/report comparisons, source manifest,
whole-phase line counts and final pytest/ruff/mypy logs. Source SHA-256 values
were unchanged throughout acceptance. R5L reference manifest was rechecked.
Output/regression_* and run_regression_suite.py are untouched.

The first MPI cProfile invocation let ranks write the same file, so its profile
was unsuitable for attribution. Repeated R3 profiling with one file per rank;
the profiling run also passes zero-tolerance output comparison. R3.prof and
R3_top20.txt now refer exclusively to rank 0, with R3_rank0..3.prof and each
worker's top-20 also retained. The original shared-file profile is labeled
unreliable. This corrected only the external profiling harness.

### R3 cProfile top 20 (rank 0, MPI)

```text
Mon Nov  2 04:36:37 2026    ../refactor_work/elegance_phase6/final/R3_rank0.prof

         24586609 function calls (24578736 primitive calls) in 17.022 seconds

   Ordered by: cumulative time
   List reduced from 2647 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      8/7    0.024    0.003   13.063    1.866 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000    6.910    6.910 <frozen runpy>:206(run_module)
        1    0.000    0.000    4.709    4.709 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/normal_points_convert.py:23(normal_points_convert)
      2/1    0.000    0.000    4.591    4.591 <frozen runpy>:65(_run_code)
        1    0.000    0.000    3.270    3.270 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.143    0.143    3.266    3.266 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:208(parse_mini_file)
    404/2    0.036    0.000    2.799    1.400 {built-in method builtins.exec}
      2/1    0.000    0.000    2.799    2.799 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    2.799    2.799 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:206(main)
      2/1    0.000    0.000    2.787    2.787 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    2.783    2.783 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.000    0.000    2.745    2.745 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/parallel/mpi.py:292(mpi_observations)
        1    0.000    0.000    2.744    2.744 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/parallel/mpi.py:219(map_tasks)
       17    0.000    0.000    2.744    0.161 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/parallel/mpi.py:175(stream_tasks)
       16    2.742    0.171    2.743    0.171 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/parallel/mpi.py:193(receive)
        2    0.121    0.061    2.587    1.294 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/frames/earth_orientation.py:379(_read_iers)
    35069    0.359    0.000    2.519    0.000 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:138(parse_mini_line)
        1    0.000    0.000    2.512    2.512 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/programs/earth_orientation.py:20(iers_c04_earth_orientation_parameter)
    58741    0.277    0.000    2.069    0.000 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/classes/time/epoch.py:68(_civil_to_jd)
    35069    0.110    0.000    1.864    0.000 /data4/wangjinghuan/deploy_bundle/llrops_project_dev/lunarops/fileio/formats/mini.py:119(_parse_launch_epoch)
```

### Phase 6 commit index

| Commit | Change | Library net | Tests net | Docs net | Total net |
| --- | --- | ---: | ---: | ---: | ---: |
| `1028bdc` | Phase 6 MPI implementation and acceptance evidence | −342 | −269 | +141 | −470 |


## Phase 7 — unified fileio (2026-11-02)

The user authorized Phase 7, including its file-format change. The source
baseline is `85624e8`; Phase 8 has not started.

### Implementation and commit

- One implementation commit: `refactor(fileio): unify records and matrix archives`.
  Library **21,373 → 20,841 (−532)** lines; tests **9,379 → 9,422 (+43)**;
  code total **−489**. Counts include tracked Python/Cython sources; no new module.
- `result_table.write_records/read_records` own the typed scalar schema, header
  mapping, units, counts, token escaping and nullable cells. Catalogs share one
  implementation; parameter names/vectors, normal points, EOP, observations and
  predictions use it. All old table/metadata call sites and tests migrated.
- Normal-equation/covariance info is typed YAML. Removed outbound payload hashing
  and checksum manifests. Matrices use one `value` column in numeric_table,
  with vector/lowerSymmetric metadata and the original lower-triangle order.
  Deleted matrix.py and moved every caller to numeric_table. No compatibility reader.
- Data contract documentation describes the new representation.

### Verification

`pytest tests -q`: **545 passed** (8.21 s); ruff clean; mypy **0 errors**, 209 files.
No .pyx/.pxi change, so the existing OpenMP build is retained as §3.1 permits.
New tests cover gzip/plain records, Python scalar types, nulls/literal tilde,
signed zero, header/count/schema/unit failures, matrix/table equivalence and
artifact-type rejection. The first implementation test run exposed a missing
prediction-merge constant import and an incorrect exception regex; both were
fixed. A catalog type annotation was corrected for mypy. Full checks reran;
no scientific values or tolerances changed (§2.10 implementation-error rule).

| Case | Ranks | Wall seconds | Required comparison |
| --- | ---: | ---: | --- |
| R1 | 4 | 11.584 | Phase 0 orbit/integrated state; Phase 3 native diagnostics: bit identical |
| R2 | 1 | 4.586 | Phase 0 orbit; Phase 3 native diagnostics: bit identical |
| R3 | 4 | 9.191 | Phase 4 numeric reference: residual bits identical, max difference 0 m |
| R4 | 1 | 23.649 | Phase 0 predictions/windows: native types and bits identical |
| R5 | 4 | 132.429 | Phase 4 numeric reference: all scientific values bit identical |

R5 rank **26**, adjustment/sigma-weight iterations **16/160**, converged.
Maximum parameter/sigma differences, relative sigma differences, residual
changes and normal/covariance/x0 differences are all **0**. lPl, counts,
parameter order and scientific report/state values also agree. Expected
provenance/path/timing exclusions include the compatibility fingerprint, since
native input bytes changed. The previously accepted state end_exclusive →
endExclusive serialization rename is normalized for the Phase 4 comparison.
An independent snapshot of Phase 6 outputs passes **52** decoded product checks;
Phase 4 comparisons pass **30** and Phase 0 R4 passes **7**. R1/R2 have separate
checks against the exact required orbit/integrated/native-diagnostic references.
R5L and R6 were not rerun: §3.2 schedules neither during Phase 7. Their existing
reference values were included in the format-preservation checks below.

### Reference products and provenance

Before editing, captured decoded products with the original readers and kept
an unchanged `pre_format/` checkout. Re-emitted reference values through the
new writers; this is an offline format migration, not a repeated physics run.
Original files, decoded pickles, original/new SHA-256 manifests and migration
scripts are preserved in ../refactor_work/elegance_phase7/.

- elegance_baseline: **101** decoded products unchanged after migration;
  frozen YAML configurations were not modified.
- elegance_phase4/numeric_reference: **52** decoded products unchanged,
  including R5L parameters/sigma/normals/covariance.
- output/regression_20261009: **432** text/structured products and **32**
  historical binary products migrated. Historical LLRMTX01 and versioned
  LUNARTBL were decoded using the historical checkout, with no production
  format branch. The two external historical orbit references contributed
  another **4** binary tables; **6** external historical residual tables were
  migrated so their scientific comparisons remain available.
- Historical normal points were reimported from the same TOTALOBS6924 source:
  every float and epoch part agrees with the archived records; only the
  already-established canonical identity case is applied. Original dataset
  counts, names and importMetadata are preserved. The first external migration
  omitted wrapper-added importMetadata; it was restored from the frozen
  decoded values and the complete preservation check rerun.
- Older report keys also predate the current consumer contract. The one-time
  migration renames **166** known static keys (e.g. sigma0_post → sigma0Post),
  using a recorded explicit mapping; no numerical value changes. Original
  report payloads remain in the decoded snapshot and original-file backups.

`scripts/run_regression_suite.py compare` now completes with exit **0**; the
version-1 normalEquationMetadata reader error is resolved. Its 14-case flags
match the archived pre-migration summary exactly (regression_flags_comparison.json).
`complete=true`, `source_equivalence_passed=true`; `passed=false` and
`current_regression_passed=false` remain visible. The sole current-regression
failure is the archived estimate_igg3 **converged=false**; six residual cases
retain their existing historical-only differences. Direct/extended estimation,
predictions and both orbit cases pass. No new failed case, hidden historical
comparison, changed convergence rule or tolerance was introduced. These known
archived failures are recorded separately from the passed Phase 7 R1–R5 gates.
The migrated text/structured, matrix and external historical arrays are also
compared to the independently decoded original values, including float bits.

### R3 cProfile top 20 (rank 0, MPI)

A separate four-rank profiling run writes one .prof per rank. Its wall time
is 19.3 s; its 494 residual records also pass the bit comparison; the main rank records 18.893 s. The unprofiled R3 timing above is
used for the regression wall clock.

```text
Mon Nov  2 04:54:06 2026    ../refactor_work/elegance_phase7/R3_rank0.prof

         30183180 function calls (30175296 primitive calls) in 18.893 seconds

   Ordered by: cumulative time
   List reduced from 2658 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      8/7    0.022    0.003   14.603    2.086 lunarops/programs/registry.py:196(run_program)
      2/1    0.000    0.000    7.252    7.252 <frozen runpy>:206(run_module)
        1    0.005    0.005    5.236    5.236 lunarops/programs/normal_points_convert.py:23(normal_points_convert)
      2/1    0.000    0.000    4.578    4.578 <frozen runpy>:65(_run_code)
        7    0.092    0.013    3.788    0.541 lunarops/fileio/result_table.py:96(write_records)
   108061    0.211    0.000    3.655    0.000 {method 'join' of 'str' objects}
  1234539    0.607    0.000    3.444    0.000 lunarops/fileio/result_table.py:110(<genexpr>)
        1    0.000    0.000    3.246    3.246 lunarops/fileio/formats/normal_point_sources.py:111(read_normal_point_source)
        1    0.145    0.145    3.243    3.243 lunarops/fileio/formats/mini.py:208(parse_mini_file)
    404/2    0.036    0.000    2.810    1.405 {built-in method builtins.exec}
      2/1    0.000    0.000    2.810    2.810 lunarops/__main__.py:1(<module>)
      2/1    0.000    0.000    2.810    2.810 lunarops/cli.py:206(main)
        1    0.003    0.003    2.810    2.810 lunarops/programs/earth_orientation.py:20(iers_c04_earth_orientation_parameter)
      2/1    0.000    0.000    2.799    2.799 lunarops/cli.py:42(cmd_run)
        1    0.000    0.000    2.796    2.796 lunarops/programs/llr_residuals.py:20(llr_residuals)
        1    0.000    0.000    2.757    2.757 lunarops/parallel/mpi.py:292(mpi_observations)
        1    0.000    0.000    2.757    2.757 lunarops/parallel/mpi.py:219(map_tasks)
       17    0.000    0.000    2.756    0.162 lunarops/parallel/mpi.py:175(stream_tasks)
       16    2.754    0.172    2.756    0.172 lunarops/parallel/mpi.py:193(receive)
  1132089    0.390    0.000    2.687    0.000 lunarops/fileio/result_table.py:61(_format_value)
```

Artifacts: ../refactor_work/elegance_phase7/ contains the run logs/results,
profiles, original/decoded reference snapshots, migration manifests, comparison
JSON, source/line counts and pytest/ruff/mypy logs. No numerical gate was
relaxed and no Phase 8 work was included.


## Phase 8 — performance (started 2026-11-02)

The user authorized Phase 8, including its labelled numerical items under §3.3.
8.8 will use the plan's NumPy-only variant; scipy remains optional. 8.9 is
feasibility analysis only. Starting commit: `3ff89e5`. Each numbered item is
committed separately; phase-end regression evidence will follow below.

### 8.1 Ocean pole-tide grid

Converted filtered string columns once with NumPy and filled each complex
component by indexed column assignment. Numeric-token filtering preserves the
original reader's accepted rows, including malformed/unmarked textual headers,
short rows, extra trailing fields, Unicode digits and underscores. Real/imaginary
assignment preserves signed zero. No new fallback or production format branch.
The full published grid's longitude/latitude and radial/north/east arrays match
the pre-change reader byte for byte. Same-process three-run medians: old
**1.004775 s**, new **0.794960 s** (−20.88%). The plan's estimated 0.3 s was not
reached. An initial regex filter was slower and was replaced before acceptance.
Two gzip/plain boundary tests cover row selection and signed-zero coefficients;
`pytest`: **547 passed**, ruff clean, mypy **0 errors**. No Cython change.
Artifacts: ../refactor_work/elegance_phase8/8_1/.
Commit subject: `perf(io): vectorize ocean pole grid assembly`.


### 8.2 CALCEPH epoch arrays and scalar cache

One `_jd_arrays` supplies all three batch methods; removed duplicate extraction
and returned-shape RuntimeError checks. The scalar query APIs remain in use by
observations and are retained. Their private LRU/RLock storage had no external
production access (grep audit); only that storage and locking were removed.
The Phase 3 snapshot-backed dynamics path does not use it. Body/target metadata
lookup caches remain. Query order and SI multiplication are unchanged.
Real DE440 comparisons over repeated and fractional TDB epochs: scalar/batch
states, accelerations and lunar angles all bit identical. R3 MPI4: **9.162 s**,
all **8** decoded products match the Phase 4 reference, including every residual
float bit. `pytest`: **547 passed** (8.12 s), ruff clean, mypy **0 errors**.
No Cython change or new test mirroring the extraction helper. Artifacts:
../refactor_work/elegance_phase8/8_2/.
Commit subject: `perf(ephemeris): consolidate epoch arrays and remove scalar LRU`.


### 8.3 Unified EOP interpolation and direct pickle

A single `interpolate(epoch)` returns five named values from stacked columns,
with one UTC/range check. Terrestrial transforms obtain those values once and
reuse them for UT1, polar motion and celestial-pole offsets. Displacements and
all test providers use the same API; the old methods/value classes are removed.
UT1-TAI storage and TAI-UTC reconstruction retain the original arithmetic across
leap seconds. Removed to/from_mpi_payload, reconstruction via __new__, and its
column-copy helper. The established sharedResources injection now carries the
provider itself through MPI pickle, with no worker file read.
The old reader oracle agrees for **1,004** C04 queries, including the 2016 leap
second, in all five float bit patterns; pickle retains samples and source.
R3 MPI4 **9.335 s**, **8** decoded products/residual bits identical to Phase 4.
`pytest`: **547 passed** (8.21 s), ruff clean, mypy **0 errors**. The first test
run found one assertion still indexing the removed payload dict; it now tests
the direct object and worker injection contract. No numerical implementation
fix or tolerance change was needed. Artifacts: ../refactor_work/elegance_phase8/8_3/.
Commit subject: `perf(eop): interpolate one snapshot and pickle providers directly`.

### 8.4 Shared per-observation ephemeris and transforms

The injected relativistic transform now owns exact-two-part-epoch state and
external-potential caches, cleared at the start of each light-time solve. Lunar
solid tides receive that transform rather than constructing one per evaluation;
Shapiro, station tides, prediction and partials reuse the same state queries.
No epoch rounding or changed potential summation was introduced. Backend state
rows are shared only at identical epochs. R3's 494 observations: backend batch
requests min/median/max **29/37/45** (previously **74/96/118**), below the 50-query
target; requested body rows **110/140/170** (previously **341/443/545**).
Serial full-observation median **10.259 ms**, previously **12.693 ms**; the phase
10 ms target is not yet met. R3 all **8** decoded products and R5 all **22**
products match Phase 4 bit for bit. R5 MPI4 wall clock **129.496 s**, unchanged
rank/iterations and converged. `pytest`: **547 passed** (8.24 s), ruff clean,
mypy **0 errors**; no Cython change. Implementation/test net **+21 lines**:
explicit injection and bounded per-NP reuse add storage but remove redundant
queries; this performance item does not yield a source reduction. Artifacts:
../refactor_work/elegance_phase8/8_4/, including independently measured before/
after query counts and decoded comparisons.
