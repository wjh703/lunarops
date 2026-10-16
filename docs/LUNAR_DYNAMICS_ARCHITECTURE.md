# LunarOps dynamics architecture

The translational subsystem now follows the same separation used by GROOPS:

* `BarycentricEarthMoonState` is the physical Earth/Moon state used at the
  program boundary. Production propagation uses only `MoonRelativeState`
  (`[M/E]`) and the prescribed Earth ephemeris.
* `GravitatingBody` stores a canonical body identifier and its gravitational
  parameter; `LunarDynamics` assigns integrated and prescribed trajectories.
  No force model relies on a positional list index.
* `ForceModel` evaluates one physical contribution as an `(N,3)` acceleration
  array. `LunarForceGroup` precompiles per-body masks; the production group fixes
  its acceleration target to Moon because Earth acceleration is prescribed.
* Gravity fields are represented by `GravityCoefficients`. LunarOps keeps
  only a `GravityField` adapter for Cartesian/body-fixed coordinates; it no
  longer reimplements spherical-harmonic Legendre evaluation.
* Production gravity fields are loaded from `gravityFields.earth` and
  `gravityFields.moon` file entries. Coefficient tables are not embedded in
  the propagation program.
* The Earth gravity-axis orientation matrix is generated from the DE440
  long-term pole model. Observation-only EOP/ITRF transforms are not accepted
  by orbit propagation.
* Every `FigureForce` declares physical interaction partners explicitly. These
  partners are independent of the per-body acceleration target mask. Moon
  figure reactions from prescribed Sun and planets are retained; Earth figure
  only evaluates its direct interaction with Moon because Earth recoil is
  already represented by the prescribed Earth trajectory.
* The lunar dynamics system builds ephemerides and rotation matrices once in a `DynamicsEpochData`.
  `ForceEvaluationContext` reuses contiguous position, velocity, and GM arrays across
  every force. Regular nodes and DOP853 startup stages are built in bounded
  synchronous batches; CALCEPH's native handle prefetch remains available.
* `MassCatalogCreate` converts configured GM values to SI and writes the native
  mass catalog consumed by propagation. Earth and Moon GM are derived together
  from DE440 GMB and EMRAT rather than entered independently. The catalog is the
  only runtime GM source; propagation selects perturbing bodies by group or ID.
* The single public `AdamsBashforthMoultonIntegrator` contains the numerical algorithm.
  Production propagation uses the published order-13 ABMD PECEC scheme at a 1/16-day
  fixed step. An eighth-order
  Dormand--Prince startup runs at one eighth of the main step and retains only the
  regular ABMD nodes. Delayed states during startup come from the configured
  ephemeris history. Stored ABMD nodes then enter a fixed-capacity history window
  that supplies local interpolation or extrapolation without copying the full
  trajectory prefix.

## Relativistic ordering

`NewtonianPointMassForce` computes the lower-order Newtonian acceleration and potential
for every body, including prescribed external sources. External/external terms
are epoch_data once per epoch; changing Earth/Moon corrector states updates only
their rows and columns. `EihPointMassForce` consumes this complete auxiliary system but
evaluates the first-order correction only for configured targets. Figure,
tide, SRP, and spin terms are separate contributions and are not silently
inserted into the EIH auxiliary term; doing so would require a separate
extended-body 1PN derivation.

The complete Newtonian auxiliary system uses a compiled symmetric pair loop.
Each pair distance is evaluated once and contributes to both endpoint
accelerations and potentials without allocating `(N,N,3)` Python/NumPy
temporaries.

Node observers receive the state and already-built epoch data at each
regular integration node. Acceleration diagnostics use this path and retain
only the modeled Moon contribution for each force group. Orbit samples and
diagnostics are separate native binary tables; a small text artifact contains
configuration, hashes, and run statistics.

`ForceEvaluationContext` holds the current barycentric body states used by all
acceleration models. `load_epoch_data()` copies prescribed states only when the
epoch_data epoch changes; the Moon row is updated from the relative integration state.

`DynamicsEpochData` and `ForceEvaluationContext` store two explicit rotation matrices:
`earth_fixed2inertial_matrix` maps Earth gravity axes to inertial axes, using
the DE430/DE440 dynamics orientation model; it is not the IERS
GCRS-to-ITRF transform used for observations. `moon_fixed2inertial_matrix`
maps lunar fixed (principal) axes to inertial LCRS axes, matching the
`LunarOrientationProvider.pa_to_lcrs_matrix` capability. It is stored without transposition;
Moon gravity uses its transpose to map inertial vectors to fixed axes.
`LunarDynamics` accepts their generating
functions as `earth_fixed2inertial_matrix_provider` and
`moon_fixed2inertial_matrix_provider`, without an intermediate provider container.

`PointMassGravityCache` stores Newtonian accelerations and potentials among
the epoch_data source bodies at one epoch. `DynamicsEpochData` and `ForceEvaluationContext`
expose it as `point_mass_gravity_cache`; Earth/Moon contributions are added
when evaluating the current state.
