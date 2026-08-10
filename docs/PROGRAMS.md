# Programs

LunarOps executes typed program chains from YAML:

```bash
python -m lunarops list-programs
python -m lunarops describe-program LlrResiduals
python -m lunarops describe-config
python -m lunarops validate config.yml
python -m lunarops run config.yml
python -m lunarops run config.yml --mpi
```

`--set name=value` overrides entries in `variables`. `validate` checks the
schema, graph, paths, and artifact types without evaluating the LLR model.
Each class choice is also validated against its registered type schema; model
categories allowed under `globals` are explicit, so a parametrization cannot
silently become a run-global object.

## Typical residual chain

```yaml
programs:
  - program: NormalPointsConvert
    inputFilesNormalPoints: [data/polac/TOTALOBS6924.DAT]
    outputFileNormalPoints: output/normalPoints.txt.gz
    outputFileImportReport: output/normalPointImportReport.txt.gz

  - program: LlrResiduals
    inputFilesNormalPoints: [output/normalPoints.txt.gz]
    outputLevel: standard
    outputFileObservationResults: output/oc.txt.gz
```

Only the converter reads external MINI or CRD. All model programs consume the
native `NormalPointFile`.

## Linear solution chain

`LlrNormalEquations` is the fused high-performance route.
`LlrObservationEquations` persists the fixed design rows for inspection and
`ObservationEquationsToNormals` consumes them. `NormalsAccumulate` aligns
structured parameter names across systems. `NormalsSolve` writes a typed
solution, covariance group, and solve report.

## Nonlinear adjustment

`LlrAdjustment` re-evaluates the observation model as parameter state changes.
It requires separate report, restart-state, solution, covariance, and final
normal-equation outputs. See
`configs/lunarops_reflector_bias_adjustment_detailed.yml`.

The complete generated contract for any program is available through
`describe-program`. `describe-config` returns the globals schema, program
choices, execution controls, artifact metadata, and a JSON Schema suitable for
an editor or GUI. The registry is authoritative for accepted keys.
