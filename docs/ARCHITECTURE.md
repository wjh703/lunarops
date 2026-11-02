# Architecture

LunarOps separates configuration, physical models, numerical execution and
scientific artifacts. Program and class registries expose the runnable contracts:

```bash
python -m lunarops list-programs
python -m lunarops describe-program LlrProcessing
python -m lunarops list-classes
python -m lunarops list-classes ephemerides
```

| Package | Responsibility |
|---|---|
| `config/` | YAML expansion, declarative schemas, factories and run resources |
| `base/` | Constants, parameter names and serialization |
| `fileio/` | Native typed ASCII/binary archives and scientific artifact adapters |
| `fileio/formats/` | External MINI/CRD import |
| `classes/` | Time, ephemerides, frames, physical models and parametrizations |
| `estimation/` | Dense linearization, nonlinear adjustment, robust weights and VCE |
| `programs/` | Program contracts, problem assembly and execution |
| `parallel/` | MPI task transport, worker resources and epoch prefetch |

## Configuration and resources

YAML loading reads data; run expansion resolves variables, loops and the run
envelope. Program schemas resolve their fields and nested class declarations
once. Public keys are lowerCamelCase and class types match exactly. Resolved
configuration contains only YAML-native values, including civil-time strings;
Epoch and physical objects are built at the assembly boundary. Scientific
fingerprints and reports use that same resolved configuration.

`RunContext` merges `shared` and `observationModel` defaults with program
choices. Its `ClassCache` owns cached model resources through an ExitStack;
contexts using an injected cache do not close the owner's resources. Assemblers
explicitly inject frames and dependencies into tides and delays. Frozen
`LunarPropagationProblem`, `ProcessingProblem` and `PredictionProblem` records separate
construction from execution.

## Observations and estimation

```text
native normal points -> ObservationResolver -> LightTimeSolver
  -> LlrObservationModel -> EquationBatch -> DenseLinearization -> adjustment
                         -> result rows -> typed observationResult archive
```

`EquationBatch` stores observation identities, epochs, one-way residuals, input
sigmas and partial blocks in aligned columns. Parametrizations consume batches,
write whole design blocks and apply updates to the catalog/model state. Row
objects are generated only for output, with Python-native scalar types.
`ObservationOptions` controls evaluation; `ObservationResultDetail` selects
standard or full result columns. One declared diagnostic field table supplies
residual and processing output.

`LightTimeSolver` uses the same outgoing leg for prediction and two-way ranging.
`UplinkSolution` contains transmit/bounce geometry; `LightTimeSolution` adds
reception and clock corrections. Exact-epoch ephemeris states and potentials
are shared within a solve. Receive displacement is evaluated for the first two
iterations and reused thereafter; station coordinates, event times and rotations
continue to use the actual epoch. Geodetic station coordinates are not cached.
EOP interpolation returns xp, yp, UT1-UTC, dX and dY together.

A fixed `DenseLinearization` supplies A, reduced observations and sigmas to
weighted normal construction. `NormalEquations` stores the absolute convention
`N x = W` and the linearization point `x0`. Matrix assembly finishes before its
cached symmetric matrix, spectrum, Cholesky factor and covariance are used.
The strict eigenvalue rank criterion rejects unobservable systems. Conditions
are computed lazily when reporting accesses them. `LlrAdjustmentResult` owns
the final normals, cofactor and residual products; published parameters are
`x0`, without applying the remaining final linearized correction.

## Dynamics and execution

[The dynamics architecture](LUNAR_DYNAMICS_ARCHITECTURE.md) describes the single
compiled RHS and accepted-node history. Python owns model composition, epoch
inputs and the ABM integration loop. Force/torque dataclasses select physical
terms; unsupported combinations raise ValueError during compilation.

MPI uses TASK/RESULT/STOP messages and task dispatch through one
`MpiRuntime`. Observation workers return column batches; rank 0 restores the
specified order and concatenates them. Shared EOP providers are pickled directly;
CALCEPH handles are opened locally by each process. Orbit propagation stays on
rank 0 while workers build time-only epoch batches. Deterministic task IDs and
bounded streaming keep serial/MPI science identical. Use one native thread per
MPI rank. Explicit `--mpi` requires at least two ranks.

## Artifact boundary

Native files declare an artifact type. ASCII products use `.txt[.gz]`; numeric
tables and matrix payloads use `.dat[.gz]`. Normal-equation and covariance
products are extensionless directories. Structured reports use typed text
headers followed by YAML mappings. Readers check their type and consumed fields;
output does not re-read or re-validate its own artifact. Scientific models and
estimators exchange typed objects, not serialized report dictionaries.
