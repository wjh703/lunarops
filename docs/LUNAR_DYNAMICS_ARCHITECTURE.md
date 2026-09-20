# LunarOps dynamics architecture

The translational subsystem now follows the same separation used by GROOPS:

* `EarthMoonBarycentricState` is the physical Earth/Moon state; it is converted
  explicitly to `EarthMoonIntegrationState` (`[EMB, M/E]`) for integration.
* `DynamicalBody` and `BodyId` identify bodies and distinguish integrated from
  prescribed trajectories.  No force model relies on a positional list index.
* `ForceModel` evaluates one physical contribution as an `(N,3)` acceleration
  array. `ForceGroup` precompiles per-body masks and has separate propagation
  and diagnostic paths, so the RHS does not allocate per-force result records.
* Gravity fields are represented by `pyshtools.SHGravCoeffs`. LunarOps keeps
  only a `GravityField` adapter for Cartesian/body-fixed coordinates; it no
  longer reimplements spherical-harmonic Legendre evaluation.
* Production gravity fields are loaded from `gravityFields.earth` and
  `gravityFields.moon` file entries. Coefficient tables are not embedded in
  the propagation program.
* The Earth dynamics frame is generated from the DE440 long-term pole model.
  Observation-only EOP/ITRF transforms are not accepted by orbit propagation.
* Every `FigureForce` declares physical interaction partners explicitly. These
  partners are independent of the per-body acceleration target mask, preserving
  reactions on Earth and Moon from prescribed Sun and planetary trajectories.
* `DynamicsSystem` prepares ephemerides and frames once in a `PreparedEpoch`.
  `DynamicsContext` reuses contiguous position, velocity, and GM arrays across
  every force, while `accelerations_batch` shares the prepared epoch across
  independent state channels.
* Fixed-capacity double-buffer prefetch prepares regular nodes and DOP853
  startup stages in bounded batches. CALCEPH access is serialized per handle;
  the GIL-free Newtonian preparation for different epochs runs in parallel.
* `MassCatalogCreate` converts configured GM values to SI and writes the native
  mass catalog consumed by propagation. Earth and Moon GM are derived together
  from DE440 GMB and EMRAT rather than entered independently. The catalog is the
  only runtime GM source; propagation selects prescribed bodies by group or ID.
* `Integrator` contains only the numerical algorithm. Production propagation uses
  the paper's order-13 ABMD PECEC scheme at a 1/16-day fixed step. An eighth-order
  Dormand--Prince startup runs at one eighth of the main step and retains only the
  regular ABMD nodes. Delayed states during startup come from the configured
  ephemeris history. Accepted ABMD nodes then enter a fixed-capacity ring buffer
  that supplies local interpolation or extrapolation without copying the full
  trajectory prefix.

## Relativistic ordering

`PointMassForce` computes the lower-order Newtonian acceleration and potential
for every body, including prescribed external sources. External/external terms
are prepared once per epoch; changing Earth/Moon corrector states updates only
their rows and columns. `EihForce` consumes this complete auxiliary system but
evaluates the first-order correction only for configured targets. Figure,
tide, SRP, and spin terms are separate contributions and are not silently
inserted into the EIH auxiliary term; doing so would require a separate
extended-body 1PN derivation.

The complete Newtonian auxiliary system uses a compiled symmetric pair loop.
Each pair distance is evaluated once and contributes to both endpoint
accelerations and potentials without allocating `(N,N,3)` Python/NumPy
temporaries.

Accepted-step observers receive the state and already prepared epoch at each
regular integration node. Acceleration diagnostics use this path and retain
only Earth, Moon, EMB, and Moon-minus-Earth projections. Orbit samples and
diagnostics are separate native binary tables; a small text artifact contains
configuration, hashes, and run statistics.

The same `DynamicsContext` is the extension point for lunar rotation and for
future force partials/variational equations.
