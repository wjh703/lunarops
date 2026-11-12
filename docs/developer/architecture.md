# Architecture

LunarOps separates configuration, physical models, numerical execution and
scientific files. This page describes how the packages fit together.

## Packages

| Package | Responsibility |
|---|---|
| `lunarops.config` | YAML loading, variable and loop expansion, declarative schemas, the class registry, the run context, scientific fingerprints |
| `lunarops.base` | constants, parameter names, station identities and intervals, serialization helpers |
| `lunarops.fileio` | native artifact readers and writers; `fileio.formats` imports CRD and MINI |
| `lunarops.classes.time` | `Epoch` and time-scale conversion |
| `lunarops.classes.ephemerides` | SPICE-kernel ephemerides (Chebyshev records read directly) and lunar orientation |
| `lunarops.classes.frames` | Earth orientation, ITRF–GCRS rotation, relativistic BCRS/GCRS/LCRS transformations |
| `lunarops.classes.delays`, `.displacement`, `.range_bias` | observation-model components |
| `lunarops.classes.observation` | normal points, catalogs, the light-time solution, the observation model, equation batches, prediction |
| `lunarops.classes.parametrization` | estimable parameter blocks |
| `lunarops.classes.dynamics` | bodies, forces, torques, deformation, the fused right-hand side, integrators, histories |
| `lunarops.estimation` | least-squares core (`lsq`), stochastic model (`stochastic`), Earth-orientation effects, constraints, the nonlinear adjustment (`adjustment`) |
| `lunarops.programs` | program specifications and implementations |
| `lunarops.parallel` | MPI task transport, worker resources, epoch prefetching |
| `lunarops._iers2010`, `_iers2010_core` | compiled IERS Conventions routines and their Python interface |
| `lunarops._dynamics_core` | compiled lunar dynamics kernel |

## From configuration to programs

```text
YAML file ─► load, merge extends ─► expand variables, loops, when ─► run plan
                                                      │
                          models ─► RunContext (class defaults, class cache, working dir, MPI runtime)
                                                      │
           for each call: program schema resolves fields and class declarations ─► program function
```

Loading reads plain data. Expansion substitutes variables, unrolls loops and
evaluates conditions; the result contains only YAML-native values, including
civil-time strings. Each program declares a `ProgramSpec`: its name, summary,
input and output `ArtifactSlot`s and a `ConfigSchema` of its fields. The schema
resolves the program fields and every nested class declaration once, fills the
defaults and rejects unknown keys. `Epoch` and physical objects are built only
at the assembly boundary inside the program. The same resolved configuration
feeds the scientific fingerprints and the reports.

`RunContext` holds the `models` defaults, which a program entry can override. Its class cache owns the instantiated model resources
through an `ExitStack`: identical declarations share one instance, and the
cache closes them at the end of the run. Assemblers inject frames and other
dependencies into tides and delays explicitly. The frozen problem records
`ProcessingProblem`, `PredictionProblem` and `LunarPropagationProblem` separate
construction from execution.

## Observations and estimation

```text
normal points ─► ObservationResolver ─► LightTimeSolver ─► LlrObservationModel
     ─► EquationBatch ─► DenseLinearization ─► normal equations ─► adjustment
                     ─► result rows ─► observationResult file
```

`EquationBatch` stores observation identities, epochs, one-way residuals,
a-priori sigmas and partial-derivative blocks in aligned columns. It is a
columnar object and cannot be iterated by row. Parametrizations consume
batches, write their whole design block and apply updates to the catalog or
model state. Row dictionaries exist only for output.

`LightTimeSolver` uses the same outgoing leg for prediction and for two-way
ranging. `UplinkSolution` holds the transmit and bounce geometry;
`LightTimeSolution` adds the reception and the clock corrections. Ephemeris
states and potentials at an exact epoch are shared within one solution. EOP
interpolation returns `xp`, `yp`, UT1−UTC, `dX` and `dY` together.

`DenseLinearization` supplies `A`, the reduced observations and the sigmas.
`NormalEquations` stores the absolute convention `N x = W` with the
linearization point `x0`. After assembly it caches the symmetric matrix, its
spectrum, the Cholesky factor and the covariance; condition numbers are
computed when a report asks for them. A rank-deficient system is rejected by an
eigenvalue criterion. `LlrAdjustmentResult` owns the final normals, cofactor
matrix and residual products.

`lunarops.estimation` has three layers; a module imports only from its own
layer or the layers above it:

| Package | Contents |
|---|---|
| `estimation.lsq` | least-squares machinery without LLR concepts: linearization, normal equations and their solution, row blocks, random-effect elimination, pseudo-observations, parameter products |
| `estimation.stochastic` | stochastic model of the observations: variance components, variance function, variance-component estimation, night sessions, robust weights, outlier diagnostics |
| `estimation.earth_orientation`, `estimation.constraints` | Earth-orientation random effects and parameter constraints, built on `lsq` |
| `estimation.adjustment` | the LLR adjustment: plan, screening, solver, results, convergence, reporting |

## Dynamics

Force and torque classes are frozen parameter dataclasses. `LunarForceGroup`
selects the active terms, and `compile_fused_rhs` builds one `FusedRhs` with
explicit force indices and a workspace in the compiled `_dynamics_core`.
`LunarDynamics.derivatives` and `LunarDynamics.terms` both use this single
implementation; unsupported combinations raise `ValueError` during
compilation.

`EpochInputs` is the picklable recipe for the time-dependent inputs of one
epoch: prescribed body states, Earth axes, the interactions between prescribed
bodies, GM drift and ring nodes. `EpochBatch` holds them as contiguous arrays
for a sequence of epochs. `SerialEpochSource` and `MpiEpochSource` build the
same arrays, each process with its own memory map of the kernels. `StepData` holds the
history-dependent tide and inertia data, prepared once per corrector step;
the kernel then fills in the candidate Moon state and evaluates forces,
torques and rotation.

`LunarNodeBuffer` stores accepted states and derivatives as double-double
pairs. `EphemerisHistory` and `DelayedHistory` provide body states from the
prescribed or the accepted trajectory; `TwoSidedHistory` joins the two arcs of
a two-sided run. `AdamsBashforthMoultonIntegrator` keeps its main loop in
Python, accumulates the state in `adams_accumulate` and calls the compiled
right-hand side. The independent Python reference
implementation of the physics lives only in the test suite.

## Parallel execution

MPI uses `TASK`, `RESULT` and `STOP` messages through one `MpiRuntime`. Rank 0
broadcasts a picklable specification of the model; workers build their own
model from it and evaluate tasks. Observation workers return column batches,
and rank 0 restores the original order and concatenates them. Deterministic
task identities and bounded streaming keep serial and MPI results identical.
Orbit propagation stays on rank 0 while workers build epoch batches.

## Artifact boundary

Native files declare their artifact type; programs declare the artifact type of
every input and output slot, which lets `validate` check a scenario's file graph
before running. Readers check the type and the fields they consume; writers do
not read back their own output. Scientific models and estimators exchange
typed objects, never serialized report dictionaries.
