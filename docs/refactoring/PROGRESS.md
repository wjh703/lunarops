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
