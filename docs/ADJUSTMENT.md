# Adjustment reference

`LlrAdjustment` estimates reflector coordinates and other registered parameter
blocks. The complete example is
`configs/lunarops_reflector_bias_adjustment_detailed.yml`.

## Observation model

For each normal point, the linearized equation is

```text
l = A * delta_x + B * delta_bias + e
```

The reported one-way a-priori sigma is the observation's relative precision.
It is never floored or rewritten. Before the adjustment, `accuracyScreening`
permanently rejects a sigma below

```text
max(minimumOneWayM, minimumFractionOfGroupMedian * groupMedian)
```

The median is computed inside the configured station, equipment-era, and
wavelength variance-component group. This screening is a validity check for
implausible accuracy claims, not robust residual rejection.

Every variance component starts with `sigmaFactor = 1`; there is no MAD
initialization. Its observation standard deviation is
`sigma0_i = sigmaFactor_g * aprioriSigma_i`.

## GROOPS-style processing steps

`adjustment.processingSteps` is an ordered list containing two step types:

- `selectParametrizations` declares the complete list of parametrization block
  IDs used by subsequent estimates. Each selection replaces the previous one.
  Unselected parameters keep their latest values and are still reduced from the
  observations.
- `estimate` runs a nonlinear least-squares adjustment using the currently
  selected blocks. It requires a unique `name` and independently controls
  `maxIterationCount`, `convergenceThreshold`,
  `convergenceThresholdByParametrizations`, `computeResiduals`, `adjustSigma0`,
  `computeWeights`, and optional per-estimate `robustWeights`.

Each key in `convergenceThresholdByParametrizations` is a selected block ID and
overrides the estimate's scalar `convergenceThreshold` for that block. An
estimate converges only when every selected block satisfies its threshold.

`adjustSigma0` and `computeWeights` require `computeResiduals: true`. When both
are enabled their coupled frozen-residual update is repeated ten times, matching
GROOPS. When only one is enabled it is evaluated once; when both are disabled
there is no sigma/weight inner update.

## Estimate iteration

Each nonlinear outer iteration with all three controls enabled follows this
fixed order:

```text
computeResiduals: solve once with the current sigmaFactors and weightFactors
  -> freeze postfit residuals e_i and redundancies r_i = 1 - h_i
  -> repeat 10 times without solving:
       adjustSigma0 from the frozen e_i/r_i and current weightFactors
       computeWeights from the frozen e_i/r_i and new sigmaFactors
  -> apply the parameter correction from the first solve
  -> relinearize; only now do the new factors affect parameter estimation
```

This intentional one-outer-iteration lag follows the GROOPS processing pattern.
The component scale is updated only when its active redundancy sum exceeds 3.
Robust weighting is applied only to observations with redundancy above 0.1;
lower-redundancy observations keep weight factor 1 because their residuals do
not contain enough independent information for an outlier decision. These are
algorithm invariants rather than configurable thresholds.

`robustWeights.model` supports `igg3` and `directRejection`. IGG3 uses `k0`
and `k1`; direct rejection uses `k0` only. There are no configurable
variance-ratio bounds, minimum component redundancy, stochastic convergence
tolerances, or stochastic iteration count.

## State inheritance and output

Every solved parameter correction is applied in full, and one iteration below
all active convergence thresholds ends the estimate immediately. Updated model
state, `sigmaFactors`, and `weightFactors` pass directly to the next processing
step. There is no separate range-bias initialization: range biases start from
their current a priori values and are estimated through the same normal-equation
flow as every other parametrization.

Intermediate estimates do not perform a separate final residual solve. Only the
last estimate recomputes the full observation set to publish residuals, normal
equations, covariance, and the final report. A fresh run initializes both factor
sets to 1; a restart restores them from the adjustment-state file before the
first processing step.

Restart state uses `sigmaFactors` and `weightFactors`. The report records which
factors were used by each parameter solve and which were produced for the next
outer iteration, processing-step selections, permanent accuracy rejections,
frozen-redundancy diagnostics,
parameter precision/correlation, and residual distributions.
