# LunarOps dynamics architecture

## State and physical composition

Production propagation integrates Moon-relative translation `[r_M/E, v_M/E]`
with prescribed Earth and external ephemerides. Optional rotation extends this
to `[r_M/E, v_M/E, q_m, omega_m, omega_c_in_mantle]` (16 components; 13 without
core rates). The relative acceleration is `a_M(model) - a_E(ephemeris)`.
`GravitatingBody` stores canonical body IDs and SI GM; the configured mass catalog
supplies Earth/Moon and perturber parameters. Gravity fields are loaded from
`gravityFields.earth` and `gravityFields.moon`.

Force and torque classes are frozen parameter dataclasses. `LunarForceGroup`
selects active terms; `compile_fused_rhs` creates `FusedRhs` with explicit force
indices and a compiled `_dynamics_core` workspace. `LunarDynamics.derivatives`
and `terms` use that single implementation. Unsupported forces, torque types,
missing physical sources and inconsistent field combinations raise ValueError.
The independent Python physics oracle lives only in `tests/reference_dynamics.py`.

Moon figure partners are explicit physical interaction sources. The Moon's
reaction to prescribed Sun/planet figure interactions is retained; prescribed
Earth acceleration already contains Earth recoil. Dynamic degree-2 coefficients
replace the static degree-2 contribution, with higher degrees unchanged.
Gravity-axis Earth orientation uses the selected DE430/DE440 long-term model,
separately from observation EOP/ITRF transforms. Lunar matrices map PA to LCRS;
field evaluation uses their transpose for inertial-to-fixed coordinates.

## Epoch inputs, steps and RHS

`EpochInputs` is the picklable recipe for prescribed states, axes, source/source
Newtonian terms, GM drift and ring nodes. `EpochBatch` holds contiguous arrays
for a sequence of epochs. Serial and MPI sources build the same arrays with
process-local CALCEPH handles. These inputs depend only on time.

`StepData` contains accepted-history-dependent tide/inertia data prepared once
per corrector step. The kernel alone fills the candidate Moon state and computes
force, torque and rotation terms. `FusedRhs.terms()` records native diagnostic
components while retaining the derivative accumulation order.

The EIH 1PN term uses the full lower-order Newtonian auxiliary accelerations
and potentials, including prescribed external/external interactions. A compiled
symmetric pair loop computes these source terms once per epoch. Corrector
candidates change the integrated Moon contribution. Figure, tide, radiation and
spin terms remain separate; adding them to the EIH auxiliary system would
require a different extended-body 1PN derivation.

## Accepted history and integration

`LunarNodeBuffer` stores accepted states and derivatives for both integration
and output. `EphemerisHistory` and `DelayedHistory` expose body states from the
prescribed or accepted/seed trajectory. Delayed translation and rotation use the
same history; Hermite state and derivative come from one local polynomial.
Trial corrector states are not committed as accepted history.

`AdamsBashforthMoultonIntegrator` keeps its main loop in Python. Production
configs use the order-13 ABMD PECEC algorithm at 1/16 day; DOP853 startup uses
smaller steps and supplies regular nodes. The shared integration loop also
serves delayed propagation. Explicit `rigidBackward` seed history precedes
forward delayed integration; positive delay, seed coverage and step restrictions
are declared in configuration. Out-of-coverage delayed rotation queries fail.
Earth-tide advanced translation queries retain their explicit extrapolation
semantics and statistics.

With `--mpi`, rank 0 integrates and workers prefetch only time-dependent inputs.
`SerialEpochSource` and `MpiEpochSource` share the epoch-batch contract. See
`jobs/rotation_de440_environment_10a_mpi.sbatch`; use one thread per rank.
An optional OpenMP build parallelizes compiled arithmetic, not Python ABM.

## Rotation and output

The mantle/core equations share total, mantle and core inertia with the orbital
degree-2 field. Quaternion kinematics use Hamilton scalar-first PA-to-inertial
rotation. DE440 geodetic precession modifies angular-momentum transport, not
quaternion kinematics. See [rotation physics and conventions](LUNAR_ROTATION_DEVELOPMENT.md)
for equations, normalization, pressure/viscous coupling and model distinctions.

Node diagnostics sample the accepted state with its existing epoch/step data.
Orbit output constructs each requested Epoch once, interpolates the trajectory,
and forms RTN/EMB columns in arrays. Orbit, acceleration diagnostics and rotation
are separate native binary tables; metadata records resolved settings, sources,
hashes and statistics. Integrated states and output interpolation do not use the
reference ephemeris to replace propagated values.
