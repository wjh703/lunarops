# Lunar orbit and rotation

`LunarOrbitPropagation` integrates the motion of the Moon relative to the
Earth and, optionally, the rotation of the lunar mantle and fluid core. The
Earth–Moon barycentre (EMB), the Sun, the planets and the minor bodies follow a
reference ephemeris; the Earth and the Moon respond to all of them and to each
other. The program writes the propagated orbit,
its difference from the reference ephemeris, an acceleration breakdown and,
with rotation, the attitude and angular velocities.

This is a forced subsystem, not a re-integration of the full planetary
ephemeris: the EMB and the other bodies stay on their reference
trajectories, while the Earth–Moon relative motion is free. The propagated Moon therefore differs from the reference
ephemeris by the effect of any model or parameter difference. The orbit output
reports this difference in radial, transverse and normal components.

Runnable scenarios:

| Configuration | Content |
|---|---|
| `configs/lunarops_orbit_de440.yml` | translation only, DE440 force model |
| `configs/lunarops_orbit_de430.yml` | translation only, DE430 force model |
| `configs/lunarops_orbit_rotation_rigid_de430.yml` | translation and rigid-mantle rotation |
| `configs/lunarops_orbit_rotation_deformed_de430.yml` | rotation with delayed deformation |
| `configs/lunarops_orbit_rotation_de440_environment_10a.yml` | ten-year DE440 orbit and rotation with core, deformation and the full body set |

## Prerequisite: the mass catalog

The GM of every body comes from a `massCatalog` file created by
`MassCatalogCreate` (see [Input data](input-data.md#inputs-for-lunar-dynamics)).
The orbit program selects the prescribed perturbers by catalog group
(`externalBodyGroups`) and body ID (`externalBodyIds`). The Earth and Moon GM
are read from the same catalog, and every model (point masses, gravity fields,
tides, torques, the barycentre) uses these values; GM values in gravity-field
file headers are ignored.

## State and frames

The integrated state is the selenocentric position and velocity of the Moon
relative to the Earth in the BCRS (ICRF axes, TDB, SI units):

\[
\mathbf{y} = [\mathbf{r}_{M/E},\ \mathbf{v}_{M/E}],\qquad
\ddot{\mathbf{r}}_{M/E} = \mathbf{a}_M - \mathbf{a}_E .
\]

The Earth and the Moon are placed about the ephemeris EMB with the catalog GM
ratio,

\[
\mathbf{r}_E = \mathbf{r}_B - \frac{GM_M}{GM_E+GM_M}\,\mathbf{r}_{M/E},\qquad
\mathbf{r}_M = \mathbf{r}_B + \frac{GM_E}{GM_E+GM_M}\,\mathbf{r}_{M/E},
\]

and every force term is evaluated on both bodies with the same model. All
Earth–Moon interactions (mutual point mass and 1PN terms, the reactions of the
Earth and Moon figures and of the Earth tide) therefore follow the integrated
state: the relative motion obeys \(-(GM_E+GM_M)\,\mathbf{r}/r^3+\dots\), and
initial conditions and dynamical parameters change the mean motion of the
orbit. Earth–Moon separations use the integrated vector itself rather than the
difference of barycentric positions.

With `lunarRotation` the state grows to 16 components (13 without a core):

\[
\mathbf{y} = [\mathbf{r}_{M/E},\ \mathbf{v}_{M/E},\ \mathbf{q}_m,\ \boldsymbol\omega_m,\ \boldsymbol\omega_c]
\]

with the mantle attitude as a Hamilton unit quaternion `q_m` (scalar first,
rotating PA axes to inertial axes), the mantle angular velocity `ω_m` and the
core angular velocity `ω_c`, both in mantle PA components (rad/s).

## Force model

Each force term is switched on by its own configuration key. `bodyForceTerms`
lists the terms applied to the Moon and the Earth; by default every configured term is used.

| Term | Enabled by | Model |
|---|---|---|
| `point_mass` | always available | Newtonian attraction of all selected bodies |
| `eih_1pn` | `includeEih: true` | Einstein–Infeld–Hoffmann first post-Newtonian point-mass acceleration (requires `point_mass`) |
| `figure_earth` | `gravityFields.earth` | Earth gravity field interacting with the `earthFigurePartners` (default: `moonFigurePartners` with the Moon in place of the Earth), with the J2 time variation `earthJ2TimeVariation` |
| `figure_moon` | `gravityFields.moon` | Moon gravity field interacting with the `moonFigurePartners` (default: the Earth and the planets) |
| `solar_j2` | `solarJ2` | solar oblateness about the IAU solar pole |
| `tide` | `earthTide` | delayed Earth tides raised by the Moon and the Sun |
| `lense_thirring` | `solarRelativistic.includeLenseThirring` | Lense–Thirring acceleration from the solar spin |
| `solar_radiation_pressure` | `solarRelativistic.includeRadiationPressure` | radiation pressure on the Moon with coefficient `moonEpsilon`; the Earth's radiation pressure is not modelled |

The EIH term uses the full Newtonian auxiliary system, including the
interactions between prescribed bodies. Each pair of bodies is evaluated once
per epoch in a compiled loop; corrector iterations only update the terms that
involve the Moon. Figure, tide, radiation and spin terms are separate from the
EIH system.

For `figure_moon`, the Moon's reaction to the interaction of its field with
each partner is integrated, including the prescribed Sun and planets, and the
Earth, when it is a partner, takes the field acceleration itself. For
`figure_earth`, the Moon takes the Earth-field acceleration directly and the
Earth takes the indirect acceleration \(-GM_j/GM_E\) times the field
acceleration of every partner \(j\) (Park et al. 2021, Eq. 29); DE440 uses the
Moon, the Sun, Mercury, Venus, Mars, Jupiter and Saturn. The Earth gravity axes follow a long-term orientation model,
`earthPoleModel` (`de440`, `de430` or `iau2006_2000a`; `ephemeris` selects
`earthDynamicsFrame`). This model is separate from the EOP-based Earth
orientation of the observation model.

Gravity fields are evaluated up to the degree of the file or `maxDegree`. The
spherical-harmonic kernel uses unnormalized Legendre intermediates, which can
overflow above roughly degree 150; lunar and Earth fields for LLR dynamics are
far below this limit.

Optional environment terms:

- `solarGmVariation`: linear drift of the solar GM.
- `kuiperBeltRing`: a ring of equal point masses in circular heliocentric
  orbits representing the Kuiper belt. Its trajectories are integrated once
  before the propagation.
- `lunarDegree2Gravity`: a time-variable lunar degree-2 field under the
  ephemeris attitude (prescribed rotation only). It replaces the static
  degree-2 terms of the Moon field; higher degrees are unchanged.

## Rotation

`lunarRotation` integrates the attitude and the angular velocities. The
moments of inertia `inertiaMoments` (A, B, C) are normalized by `M R²` with
`R = radiusM`, which must equal the radius of the Moon gravity field.

**Kinematics.**

\[
\dot{\mathbf{q}}_m = \tfrac12\, \mathbf{q}_m \otimes (0, \boldsymbol\omega_m).
\]

**Mantle.** With the mantle inertia `K_m = K_T − K_c` (total minus core),

\[
\mathbf{K}_m \dot{\boldsymbol\omega}_m = \mathbf{N}_\text{ext} + \mathbf{N}_\text{cmb}
- (\boldsymbol\omega_m - \boldsymbol\nu_{gp}) \times (\mathbf{K}_m \boldsymbol\omega_m)
- \dot{\mathbf{K}}_m \boldsymbol\omega_m .
\]

**Core.** For an axisymmetric core fixed in the mantle frame with constant
inertia `K_c = α_c C_T diag(1 − f_c, 1 − f_c, 1)` (`polarMomentFraction` α_c,
`flattening` f_c),

\[
\mathbf{K}_c \dot{\boldsymbol\omega}_c = -\mathbf{N}_\text{cmb}
- (\boldsymbol\omega_m - \boldsymbol\nu_{gp}) \times (\mathbf{K}_c \boldsymbol\omega_c),
\qquad
\mathbf{N}_\text{cmb} = \kappa (\boldsymbol\omega_c - \boldsymbol\omega_m)
+ (K_{c,zz} - K_{c,xx})\, \omega_{c,z}\, (\mathbf{e}_z \times \boldsymbol\omega_c),
\]

where `κ = viscosityPerS` is the viscous coupling and the second term the
pressure coupling at the core–mantle boundary.

**Geodetic precession.** With `geodeticPrecession: true`, the transport terms
use `ω_m − ν_gp`, where

\[
\boldsymbol\nu_{gp} = \mathbf{R}^\mathsf{T} \frac{3}{2c^2} \sum_{j\in\{E,S\}}
\frac{GM_j\, (\mathbf{r}_{jM} \times \mathbf{v}_{jM})}{r_{jM}^3}
\]

in PA components (Park et al. 2021, Eq. 44). The quaternion kinematics use
`ω_m` unchanged.

**External torques.** `sources` lists the bodies whose gravity acts on the
lunar figure. `gravityTorque: quadrupole` uses the inertia tensor
(`3 GM_j / r_j³ · u_j × (K u_j)`); `figure` uses the full Moon gravity field,
evaluated in the same compiled kernel and with the same coefficients as the
`figure_moon` force. `earthFigureTorque: true` adds the torque from the Earth's
J2 on the lunar figure.

**Deformation.** `lunarRotation.deformation` adds the delayed tidal and
rotational deformation of the mantle (Love number `loveK2`, delay
`delayDays`). The inertia perturbations

\[
\Delta\mathbf{K}_\text{tide} = -k_2 \frac{GM_E}{GM_M}\Big(\frac{R}{r}\Big)^3 \Big(\hat{\mathbf u}\hat{\mathbf u}^\mathsf{T} - \tfrac13 \mathbf I\Big),
\qquad
\Delta\mathbf{K}_\text{spin} = \frac{k_2 R^3}{3\,GM_M}\Big(\mathbf w \mathbf w^\mathsf{T}
- \tfrac{\mathbf w\cdot\mathbf w - n^2}{3}\mathbf I - n^2 \mathbf e_z \mathbf e_z^\mathsf{T}\Big)
\]

are evaluated with the Earth direction `u` and the mantle angular velocity `w`
at the delayed epoch `t − τ`; `n = meanMotionRadS`. Their time derivatives are
computed analytically from the same delayed history. The deformed total
inertia also defines the Moon's degree-2 gravity coefficients, so the orbit
and the rotation see one consistent figure. The static coefficients
`c21Static`, `s21Static`, `s22Static` are added to the deformed field.

Delayed states before the start epoch come from a seed history:
`seedModel: rigidBackward` integrates the rigid Moon backwards over
`seedDurationSeconds`, which must cover the delay; the step must not exceed the
delay. A delayed query outside the covered history is an error.

The equations follow Folkner et al. (2014) and Park et al. (2021); see the
[bibliography](../reference/references.md).

## Integration

Translation and rotation are integrated together with a fixed-step
Adams–Bashforth–Moulton method:

| Option | Default | Meaning |
|---|---|---|
| `stepSeconds` | 5400 s (1/16 day) | fixed step |
| `integratorOrder` | 13 | method order |
| `correctorIterations` | 2 | corrector passes (PECEC) |
| `startupStepSeconds` | 675 s | step of the DOP853 start-up |
| `historyInterpolationOrder` | 13 | interpolation order for delayed states |
| `trajectoryInterpolationOrder` | 13 | interpolation order for output |

The first `order − 1` nodes are produced by an eighth-order Dormand–Prince
(DOP853) start-up with the smaller `startupStepSeconds`. Accepted nodes and
their derivatives are stored; trial corrector states never enter the accepted
history. Delayed states are interpolated locally from this history with one
Hermite polynomial that gives the state and its derivative together; output
epochs are interpolated locally between nodes.
The propagation may run backwards (negative `durationSeconds`), and the
duration must be a whole multiple of the step.

The default start-up step comes from one-year runs of the full 384-body
model. Compared with 675 s, start-up steps of 1350 s and 5400 s change the
Moon–Earth position by up to 1.3 mm and 1.6 mm, above the 1 mm criterion, so
675 s is the default. The start-up step only affects the fixed start-up cost,
not the cost per node of a long arc.

## Outputs

All tables are binary `LUNARTBL` files (`.dat`) with named columns; the
metadata is a typed text file.

`outputFileOrbit` (`lunarOrbit`)
:   Per output epoch (`outputStepSeconds`), the offset from the start in TDB
    seconds and the barycentric positions and velocities of the Earth, Moon and
    Earth–Moon barycentre, the relative state, and the difference from the
    reference ephemeris in radial, transverse and normal components (m) and in
    velocity (m/s).

`outputFileAccelerationDiagnostics` (`lunarAccelerationDiagnostics`)
:   Every `accelerationDiagnosticsStepSeconds` (a multiple of the step), the
    Moon and Earth accelerations split into the groups `newtonian`,
    `post_newtonian`, `figure` (solar J2, Earth figure, Moon figure), `tide`,
    `srp` and `lt`, in m/s² (columns `<group>_moon_*` and `<group>_earth_*`).
    The integrated acceleration is the Moon sum minus the Earth sum.

`outputFileRotation` (`lunarRotation`)
:   Quaternion, mantle and core angular velocities, the quaternion norm error,
    every torque term, the core–mantle viscous and pressure torques, the
    geodetic precession rate, the angular accelerations and, with deformation,
    the total and mantle inertia and the inertia rate.

`outputFileMetadata` (`lunarOrbitMetadata`)
:   The resolved configuration, the active force inventory, the gravity-field
    sources (path, normalization, tide system, degrees, missing coefficients,
    GM and radius sources), input checksums and run statistics.

Quaternions in the output are sign-aligned between neighbours and normalized;
the integrated state itself is not modified.

## Parallel and threaded execution

With `--mpi`, rank 0 integrates while the workers precompute the
time-dependent inputs of future epochs (ephemeris states, Earth axes and the
interactions between prescribed bodies). The results are identical to a
serial run. An OpenMP build parallelizes the compiled body-pair loop. See
[Parallel execution](parallel-execution.md).
