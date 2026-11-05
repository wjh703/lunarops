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

By default a block covers every observed reflector or station that is in the
catalog; `reflectors`/`stations` restrict it and `fixedStations` holds stations
at their catalog positions. `empiricalLibration` terms rotate the PA frame about
the PA axes (`x`, `y`, `psi`) or as physical librations referred to the J2000
mean ecliptic (`rho`, `Isigma`, `tau`). Their argument is an integer combination
of the Delaunay arguments or a period in days. They enter the observation
equations through the orientation partials; terms with `estimate: false` are
applied with their given amplitudes.

The class options are listed in the
[model class reference](../reference/classes.md#parametrization).

## Processing steps

`processingSteps` is an ordered plan:

1. exactly one `screenObservations` step, first;
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
2. Weight each observation with `p = w / (s σ)²`, where `σ` is its a-priori
   one-way sigma, `s` the sigma factor of its variance component and `w` its
   robust weight factor.
3. Form and solve the normal equations `N x = Aᵀ P l`. A rank-deficient system
   is an error: LunarOps checks the eigenvalue spectrum of `N` and does not
   regularize.
4. If `computeResiduals` is true, compute the postfit residuals and
   redundancies `rᵢ = 1 − pᵢ aᵢᵀ N⁻¹ aᵢ` and update the stochastic model on
   these frozen residuals:
   - `estimateVarianceFactors`: each component's new sigma factor is
     `sqrt(Σ wᵢ (vᵢ/σᵢ)² / Σ rᵢ)` over its active observations; a component
     with a total redundancy of 3 or less keeps its factor.
   - `estimateRobustWeights`: each observation's weight factor follows from its
     standardized residual `vᵢ / (s σᵢ √rᵢ)` and the `robustWeighting` function.
   With both enabled, the two updates alternate ten times per iteration.
5. Apply the parameter update. The step converges when the largest update of
   every parameter block is at most `convergenceThreshold` (per block:
   `convergenceThresholdByParametrizations`), or stops after
   `maxIterationCount` iterations.

Robust weighting functions, for a standardized residual `|z|`:

| Model | `|z| ≤ k0` | `k0 < |z| ≤ k1` | `|z| > k1` |
|---|---|---|---|
| `igg3` | 1 | `(k0/|z|) ((k1 − |z|)/(k1 − k0))²` | 0 |
| `directRejection` | 1 | — | 0 (above `k0`) |

Sigma factors and weight factors carry over from one estimate step to the
next. A fresh run starts with all factors equal to 1.

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
    and updated reflector or station catalogs.

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
