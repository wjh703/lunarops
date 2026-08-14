# LLR processing reference

`LlrProcessing` estimates reflector coordinates and other registered parameter
blocks. See `configs/lunarops_reflector_bias_adjustment_detailed.yml` for the
complete configuration.

## Processing steps

`processingSteps` is a required top-level program field. Supported steps are:

- `screenObservations`: permanently define the observation domain from initial
  absolute residual and reported-sigma criteria. It must be the first step and
  may occur only once.
- `selectParametrizations`: replace the active block list for later estimates.
- `estimate`: run nonlinear least squares with a unique `name` and optional
  iteration, convergence, residual, sigma-factor, and robust-weight controls.
- `writeResiduals`: write final postfit residuals at `standard` or `full` level.
- `writeNormalEquations`: write the final normal equations.
- `writeResults`: write one or more of report, restart state, solution,
  covariance, and updated reflector catalog.

All selection and estimate steps precede output steps. Output steps consume the
last estimate, which performs the final full residual evaluation.

## Stochastic model

The normal-point one-way a-priori sigma is
`0.5 * c * uncertainty_two_way_s`. `screenObservations.reportedSigma` rejects
implausibly small reported sigmas without changing or flooring retained values.
`screenObservations.residual` rejects excessive initial absolute O-C values.
Every configured variance component starts with `sigmaFactor = 1`, and exactly
one component must match every retained observation.

When `estimateVarianceFactors` and `estimateRobustWeights` are both enabled in
an `estimate`, their coupled update is repeated ten times on frozen postfit
residuals and redundancies, following the GROOPS processing pattern.
`robustWeighting.model` supports `igg3` (`k0`, `k1`) and `directRejection`
(`k0` only).

## State and outputs

Solved parameter updates, `sigmaFactors`, and `weightFactors` pass directly to
the next estimate. A fresh run initializes both factor sets to one. Set
`inputFileProcessingState` to restore model, sigma, and weight state; its
scientific fingerprint must match the current input and model configuration.

`writeResults.outputFileState` uses the `processingState` artifact, while
`outputFileReport` uses `processingReport`. There is no standalone apply-
solution program: the final reflector catalog is written directly by
`writeResults.outputFileReflectorCatalog`.
