# Parameter estimation

`LlrProcessing` estimates model parameters from normal points by nonlinear
weighted least squares. The parameters, the weighting and the order of the
work are all declared in the configuration; `configs/lunarops_reflector_bias_adjustment_detailed.yml`
is a complete example.

```yaml
- program: LlrProcessing
  inputFilesNormalPoints: ["{outputDir}/normal_points.txt"]
  inputFileStationCatalog: "{outputDir}/stations.txt"
  inputFileReflectorCatalog: "{outputDir}/reflectors.txt"
  parametrization:
    - {type: reflectorPosition}
    - {type: stationRangeBias, per: station}
  varianceComponents:
    - {id: apollo, station: APOLLO, start: 2006-01-01, endExclusive: null}
    - {id: grasse, station: GRASSE, start: 1984-01-01, endExclusive: null}
  processingSteps:
    - type: screenObservations
      residual: {maximumAbsoluteM: 2.0}
      reportedSigma: {minimumOneWayM: 1.0e-3, minimumFractionOfGroupMedian: 0.1}
    - type: estimate
      name: joint
      maxIterationCount: 10
      robustWeighting: {model: igg3, k0: 1.5, k1: 6.0}
    - type: writeResiduals
      outputFile: "{outputDir}/postfit.txt"
      outputLevel: full
    - type: writeResults
      outputFileReport: "{outputDir}/processing_report.txt"
      outputFileState: "{outputDir}/processing_state.txt"
      outputFileSolution: "{outputDir}/solution.txt"
      outputFileCovariance: "{outputDir}/covariance"
      outputFileReflectorCatalog: "{outputDir}/reflectors_estimated.txt"
```

The program also takes every observation-model field of `LlrResiduals`
(ephemeris, Earth rotation, delays, displacements, range bias, time window,
elevation limits); see the [program reference](../reference/programs.md#llrprocessing).

## Parametrizations

`parametrization` lists the parameter blocks that estimate steps can use:

| Type | Parameters | Unit |
|---|---|---|
| `reflectorPosition` | X, Y, Z of each reflector in the Moon PA frame | m |
| `stationPosition` | X, Y, Z of each station in the ITRF, optionally the velocity | m, m/yr |
| `stationRangeBias` | one-way range bias per station, or per station and interval | m |
| `empiricalLibration` | cosine and sine amplitudes of periodic lunar orientation terms | mas |
| `lunarDynamics` | corrections to initial values and dynamical parameters of the integrated lunar orbit and rotation | unit of each quantity |
| `lunarLoveNumber` | correction to the lunar displacement Love number h2 | 1 |

By default a block covers every observed reflector or station that is in the
catalog; `reflectors`/`stations` restrict it and `fixedStations` holds stations
at their catalog positions. `empiricalLibration` terms rotate the PA frame about
the PA axes (`x`, `y`, `psi`) or as physical librations referred to the J2000
mean ecliptic (`rho`, `Isigma`, `tau`). Their argument is an integer combination
of the Delaunay arguments or a period in days. They enter the observation
equations through the orientation partials; terms with `estimate: false` are
applied with their given amplitudes.

With the Earth-orientation parameters fixed to a tabulated series, the LLR data
determine the origin and orientation of the station network, so every station
can be estimated and none needs to be fixed; fixing a station imposes its
catalog coordinates on the solution. Station coordinates in a given frame (for
example ITRF2020, whose SLR stations include the LLR telescopes at Grasse,
Matera, Wettzell and McDonald but not APOLLO) follow from `minimalConstraints`
on stations with coordinates in that frame (see
[constraints](../reference/programs.md#constraints-entries)), which are needed
once Earth-orientation offsets are estimated.

The class options are listed in the
[model class reference](../reference/classes.md#parametrization).

### Lunar orbit and rotation: `lunarDynamics`

`lunarDynamics` estimates corrections to the quantities of a
[`lunarDynamicsPartials`](../reference/file-formats.md#lunar-dynamics-partials)
table: the initial position, velocity, Euler angles and angular velocities of
the integrated Moon and its dynamical parameters (see
[lunar dynamics](lunar-dynamics.md#variational-equations)). The observation
model must use the ephemeris of the same integration.

```yaml
parametrization:
  - type: lunarDynamics
    inputFilePartials: "{orbitDir}/lunar_dynamics_partials.dat"
    inputFileLunarDynamicsParameters: "{orbitDir}/lunar_dynamics_parameters.txt"
    quantities: [r0_x, r0_y, r0_z, v0_x, v0_y, v0_z, gm_emb, beta, gamma]   # default: all of the table
    interpolationOrder: 9
```

For a normal point with bounce epoch `t_b` (TDB), the design row of quantity
`x` is the chain rule

```text
∂ρ/∂x = ∂ρ/∂r · ∂r/∂x(t_b) + ∂ρ/∂w · E(e(t_b)) · ∂e/∂x(t_b)
```

- `∂ρ/∂r = (û_up + û_down)/2` is the partial of the one-way range with respect
  to the Moon-minus-Earth position at a fixed Earth–Moon barycentre, with the
  BCRS unit vectors from the station to the reflector on the two legs. The
  reflector moves by `GM_E/(GM_E + GM_M) δr` and the stations by
  `−GM_M/(GM_E + GM_M) δr`, so the reflector moves by `δr` relative to the
  station.
- `∂ρ/∂w` is the partial with respect to a small rotation `w` of the PA frame,
  the one `empiricalLibration` uses, and `E` maps increments of the 3-1-3 Euler
  angles `e = (φ, θ, ψ)` to that rotation:

  ```text
      ⎡ sin θ sin ψ    cos ψ   0 ⎤
  E = ⎢ sin θ cos ψ   −sin ψ   0 ⎥
      ⎣ cos θ           0      1 ⎦
  ```

- `∂r/∂x` and `∂e/∂x` are the position and Euler-angle rows of the table,
  interpolated to `t_b` with a Lagrange polynomial of order
  `interpolationOrder` (ten nodes by default) on its regular grid.

The change of the event epochs with `x` is neglected, a relative error of
about `v/c ≈ 1e-6` of the column, as for the other partials. Every bounce epoch
must lie inside the table; otherwise the program stops.

The parameters `moon:dynamics.<quantity>` are corrections to the nominal values
of the integration, in the unit of the quantity, and start at zero. The
observation model is not re-evaluated with them: the block subtracts
`design · correction` from O−C. A correction is therefore exact only to first
order, and the fit is iterated with the integration: integrate, estimate, write
the updated parameters with `outputFileLunarDynamicsParameters` of
`writeResults` (nominal value plus correction, at the same epoch), integrate
again from that file, until the corrections are negligible against their
formal errors. Without `inputFileLunarDynamicsParameters` the nominal initial
values are read from the table at offset 0, the nominal dynamical parameters
are unknown and the updated file cannot be written for them.

The update of this block that is compared with `convergenceThreshold` is the
largest change of the range it causes, in metres: each correction times the
root mean square of its design column. A-priori sigmas are set with the
`aprioriSigma` constraint on the correction, in the unit of the quantity, for
example `parameters: ["moon:dynamics.gm_emb::"]`.

The design columns differ by many orders of magnitude (about 1 m per metre of
`r0_x`, 1e12 m per rad/s of `wm0_z`, 1e-6 m per m³/s² of `gm_emb`). The normal
equations are therefore factorized, and their rank and condition number
evaluated, after scaling to a unit diagonal (`D⁻¹ N D⁻¹` with
`D = diag(N)^½`); the reported condition number is that of the scaled matrix
and does not depend on the units of the parameters.

### Lunar Love number: `lunarLoveNumber`

`lunarLoveNumber` estimates a correction to the displacement Love number `h2`
of the `lunarSolidTide` reflector displacement, which must be configured. The
displacement is linear in `h2`, so the block is exact: its design column is the
range gradient with respect to the reflector position times the displacement of
unit `h2`, and `column · correction` is subtracted from O−C. The state reports
the nominal `h2` of the model, the correction and their sum; to apply the
estimate in a later run, set `reflectorDisplacement.h2` to the updated value.

## Processing steps

`processingSteps` is an ordered plan:

1. a `screenObservations` step, first; a later one screens again with the residuals of the estimates before it
   (useful with `lunarDynamics`, whose prefit residuals can be metres);
2. any number of `selectParametrizations` and `estimate` steps (at least one
   `estimate`);
3. one or more output steps: `writeResiduals`, `writeNormalEquations`,
   `writeResults`.

Output steps report the state after the last `estimate`.

### `screenObservations`

Screening fixes the observation domain once, before estimation. From the
light-time-converged normal points it removes:

- normal points whose absolute prefit one-way residual exceeds
  `residual.maximumAbsoluteM` (or the per-station value in
  `residual.maximumAbsoluteByStationM`);
- normal points whose reported one-way sigma is implausibly small: below
  `reportedSigma.minimumOneWayM` or below `minimumFractionOfGroupMedian` times
  the median sigma of their variance component, whichever limit is larger.

Retained sigmas are used as reported; they are never floored or rescaled by
the screening.

### `selectParametrizations`

Replaces the set of active parameter blocks for the following estimate steps,
for example to estimate reflector positions first and add range biases later.
The values of inactive blocks stay at their current values.

### `estimate`

Each estimate step runs a Gauss–Newton iteration:

1. Evaluate the full observation model at the current parameter values and
   linearize: design matrix `A` and reduced observations `l`.
2. Weight each observation with `p = w / σ²`, where `w` is its robust weight
   factor and `σ` its one-way sigma from the variance model of its component:
   `σ = s σ_rep` (`varianceModel: scaled`) or `σ² = a² + (s σ_rep)²`
   (`floorAndScaled`), with reported sigma `σ_rep`, sigma factor `s` and
   floor `a`. With `sessionCorrelation`, each night session also gets a random
   offset of sigma `τ`; with `earthOrientationCorrection`, each station night
   gets random corrections of UT1 and the polar motion. Both are eliminated
   from the normal equations.
3. Form and solve the normal equations `N x = Aᵀ P l`. A rank-deficient system
   is an error: LunarOps checks the eigenvalue spectrum of `N` and does not
   regularize.
4. If `computeResiduals` is true and `estimateVarianceFactors` or
   `estimateRobustWeights` is set, solve and reweight until the stochastic
   model is stable (at most 100 passes per iteration):
   - `estimateVarianceFactors`: the variance parameters `s`, `a` and `τ` of each
     component are estimated from the residuals (least-squares variance
     component estimation), divided by the Gaussian consistency factor of the
     robust weight function so that they stay unbiased under robust weights. A
     component with a total redundancy of 3 or less keeps its parameters.
   - `estimateRobustWeights`: each observation's weight factor follows from its
     residual standardized by its variance under the current weights (Baarda's
     w-statistic) and the `robustWeighting` function.
   The report flags an iteration whose passes reached the limit
   (`stochasticModelConverged`).
5. Apply the parameter update. The step converges when the largest update of
   every parameter block is at most `convergenceThreshold` (per block:
   `convergenceThresholdByParametrizations`), or stops after
   `maxIterationCount` iterations.

Robust weighting functions, for a standardized residual `|z|`:

| Model | `|z| ≤ k0` | `k0 < |z| ≤ k1` | `|z| > k1` |
|---|---|---|---|
| `igg3` | 1 | `(k0/|z|) ((k1 − |z|)/(k1 − k0))²` | 0 |
| `directRejection` | 1 | — | 0 (above `k0`) |

Sigma factors, floors, session sigmas and weight factors carry over from one
estimate step to the next and are saved in the restart state. A fresh run
starts with all factors equal to 1, zero floors and session sigmas of
`sessionInitialSigmaM`.

## Variance components

`varianceComponents` divides the observations into groups with their own sigma
factor. A component matches the normal points of one station whose transmit
date lies in `[start, endExclusive)` and, optionally, whose wavelength lies in
`[wavelengthMinNm, wavelengthMaxExclusiveNm)`. Every retained normal point must
match exactly one component.

## Outputs

`writeResiduals`
:   Postfit residuals as an `observationResult` table. The `standard` level has
    the observation identity, epoch, station, the current-state and linearized
    postfit residuals, the residual sigma, the standardized residual and the
    applied weight factor; `full` adds every diagnostic of the report.

`writeNormalEquations`
:   The final normal equations as a directory, stored in the absolute form
    `N x = W` with the linearization point `x0`: `info.txt` (observation and
    parameter counts, `lPl` at `x0`), `metadata.txt` (scientific conventions),
    `parameterNames.txt`, and the binary matrices `normalMatrix.dat.gz` (lower
    triangle), `rightHandSide.dat.gz` and `x0.dat.gz`. Combining such
    directories requires matching conventions.

`writeResults`
:   Any of: the processing report (`outputFileReport`), the restart state
    (`outputFileState`), the parameter vector with formal uncertainties
    (`outputFileSolution`), the covariance directory (`outputFileCovariance`),
    updated reflector or station catalogs, and the updated lunar dynamics
    parameters of a `lunarDynamics` parametrization
    (`outputFileLunarDynamicsParameters`).

The published parameter values are those of the last linearization point;
the remaining linearized correction of the final iteration is reported but not
applied.

## Processing report and publication quality

The report contains the settings, a summary, per-parameter values and
uncertainties, per-observation results, the sigma factors and weight-factor
statistics of each iteration, and a `publicationQuality` section. A result is
eligible as a formal product (`formalProductEligible: true`, `status: passed`)
only if the adjustment converged, the normal equations have full rank, the
posterior redundancy is positive and `sigma0Post` is defined. Otherwise the
status is `diagnostic` and `reasons` lists why, for example the termination
reason of a non-converged run, clamped tropospheric delays or failed
light-time solutions.

## Continuing a run

`writeResults.outputFileState` stores the parameter values, sigma factors,
weight factors and the scientific fingerprint. A later run that sets
`inputFileProcessingState` starts from these values. Its fingerprint (inputs,
models, parametrization and processing plan) must match; otherwise the run is
rejected.
