# LunarOps

LunarOps is a processing system for Lunar Laser Ranging (LLR). It takes laser
normal points from the observing stations and turns them into scientific
products: observed-minus-computed residuals, estimated reflector, station,
range-bias and libration parameters with their covariance, observation
predictions for upcoming campaigns, and numerically integrated lunar orbits
and rotations.

## What it does

| Task | Programs | Guide |
|---|---|---|
| Import normal points, EOP and coordinates into typed native files | `NormalPointsConvert`, `StationCatalogCreate`, `ReflectorCatalogCreate`, `MassCatalogCreate`, `IersC04EarthOrientationParameter`, `IersRapidEarthOrientationParameter`, `EarthOrientationParameterMerge` | [Input data](user-guide/input-data.md) |
| Compute O−C residuals with the full two-way light-time model | `LlrResiduals` | [Observation model](user-guide/observation-model.md) |
| Estimate parameters by nonlinear weighted least squares | `LlrProcessing` | [Parameter estimation](user-guide/estimation.md) |
| Predict pointing and visibility windows for a station and reflector | `LlrObservationPrediction`, `LlrObservationPredictionMerge` | [Observation prediction](user-guide/prediction.md) |
| Integrate the lunar orbit, optionally coupled with mantle and core rotation | `LunarOrbitPropagation` | [Lunar orbit and rotation](user-guide/lunar-dynamics.md) |

## Design in brief

- **One configuration file per run.** A YAML file lists the programs to run,
  in order, with their inputs, outputs and model choices. Variables and loops
  keep scenarios compact, and `validate` checks a whole scenario before
  anything is computed. See [Configuration](user-guide/configuration.md).
- **Strict, explicit contracts.** Every configuration key, class type, program
  name and catalog key must match exactly. Unknown keys fail. Time scales,
  frames and units are part of each interface.
- **Typed native files.** Every product begins with a header that names its
  artifact type, and readers check that type and the fields they use. See
  [File formats](reference/file-formats.md).
- **Reproducible science.** Results carry a scientific fingerprint of their
  inputs and models. Serial, worker-process and MPI runs produce the same
  numbers.
- **Compiled where it counts.** The lunar dynamics right-hand side and the
  IERS routines are compiled Cython extensions. Model composition, I/O and
  estimation are Python and NumPy.

## Where to start

1. [Install LunarOps](getting-started/installation.md).
2. Follow the [quickstart](getting-started/quickstart.md) from raw normal points
   to residuals.
3. Look up any program option in the [program reference](reference/programs.md)
   and any model class in the [model class reference](reference/classes.md).

Developers extending LunarOps should read the
[architecture overview](developer/architecture.md) and
[Extending LunarOps](developer/extending.md).
