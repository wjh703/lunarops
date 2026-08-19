# Architecture

LunarOps uses boundaries inspired by GROOPS: configuration selects classes and
programs, typed objects carry data between layers, and each program owns one
complete processing task.

```text
config/       registry, YAML loader, run context and shared object cache
base/         constants, parameter names and validation helpers
fileio/       typed native artifact readers/writers and low-level encodings
  formats/    external CRD/MINI adapters and source dispatch
classes/      time, ephemerides, frames, delays, displacement, observation
              and parametrization implementations
estimation/   nonlinear adjustment, robust weights, VCE and least squares
programs/     independently selectable processing tasks
parallel/     MPI transport and worker lifecycle
```

## Runtime flow

```text
typed normal-point file
  -> NptRecord
  -> ObservationResolver
  -> LightTimeSolver
  -> LlrObservationModel
  -> ObservationEquation
  -> residual table or nonlinear processing
```

`ObservationEquation` is the estimation contract. It contains the one-way
residual, input-derived sigma, identity keys, epoch, and named partial blocks.
Typed observation-result and report artifacts are created at the output
boundary; estimators do not reconstruct equations from output dictionaries.

Canonical normal points, station/reflector records, and catalog identity
resolution live under `classes/observation/`; catalog coordinates arrive from
program inline fields or native files.
`fileio/` owns native artifact representations and low-level encodings.
`fileio/formats/` translates CRD/MINI sources only at the import boundary;
normal-equation arithmetic lives under `estimation/`.

`Parametrization` blocks declare named columns, provide design entries, and
absorb solved updates into model state. `LlrProcessing` relinearizes after
updates and can publish residuals, normal equations, solution, covariance,
restart state, and the updated reflector catalog as explicit output steps.

## Extension points

To add a physical model, implement the typed interface in the relevant
`classes/` category, register a `ConfigSchema` with the factory, mark it
`global_scope=True` only when it is valid under `globals:`, and add focused
tests for units, signs, and reference values. To add an estimable quantity,
implement a `Parametrization`, register its schema, and provide its named
partial block. The solver, normal-equation format, and CLI then remain
unchanged.

## Resource and parallelism rules

`RunContext` owns the class instances and transient resources it creates in its
local cache. An injected cache is shared state and is non-owning by default;
the caller closes it once after all child contexts finish. MPI workers
construct process-local handles after their initialization barrier, and native
handles are never serialized between ranks. `mpi.chunksize` controls task
granularity. The removed local `ProcessPoolExecutor` and legacy
`LlrReflectorFit` paths are not supported.
