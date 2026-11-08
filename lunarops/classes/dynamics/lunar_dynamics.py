"""Right-hand side of the Moon-minus-Earth translation and the lunar rotation.

The Earth-Moon barycentre (EMB) and every external body are prescribed by the
ephemeris; the Earth-Moon relative motion is propagated. The integration
state is

- ``y[0:3]``: Moon-minus-Earth position in m, BCRS with ICRF axes;
- ``y[3:6]``: Moon-minus-Earth velocity in m s⁻¹;
- ``y[6:10]``: scalar-first quaternion from Moon PA to inertial axes;
- ``y[10:13]``: mantle angular velocity in PA axes, rad s⁻¹;
- ``y[13:16]``: fluid-core angular velocity in PA axes, rad s⁻¹;

with rows 6–15 present only when rotation (and a core) is integrated. Time
is TDB seconds. The Earth and the Moon are placed about the EMB,

    r_E = r_B - GM_M / (GM_E + GM_M) y,    r_M = r_B + GM_E / (GM_E + GM_M) y,

and the translational derivative is a_M - a_E, both accelerations evaluated
with the same force model at these positions. Every Earth-Moon interaction
(the mutual point-mass and 1PN terms, the figure and tide reactions) therefore
follows the integrated state, so initial conditions and dynamical parameters
change the mean motion of the propagated orbit instead of being pulled back to
the reference lunar orbit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.time import Epoch

from .bodies import EMB, GravitatingBody
from .context import StepData
from .epoch_data import EphemerisLunarAxes, EpochBatch, EpochInputs, OrientationProvider
from .force_models import LunarForceGroup
from .fused_rhs import compile_fused_rhs
from .rotation import LunarRotationDynamics
from .solar_environment import KuiperBeltRing, SolarGmVariation


class LunarDynamics:
    """Lunar dynamics model compiled into the native right-hand side.

    Parameters
    ----------
    earth_body : GravitatingBody
        The Earth.
    moon_body : GravitatingBody
        The Moon.
    perturbing_bodies : sequence of GravitatingBody
        External bodies prescribed by the ephemeris.
    ephemeris : Ephemeris
        Source of the prescribed BCRS states of the EMB and the external
        bodies.
    force_group : LunarForceGroup or None, optional
        Force models acting on the Moon and the Earth; none when ``None``.
    earth_fixed2inertial_matrix_provider : OrientationProvider or None, optional
        Earth gravity-field axes; identity when ``None``.
    moon_fixed2inertial_matrix_provider : OrientationProvider, EphemerisLunarAxes or None, optional
        Prescribed lunar PA axes for orbit-only propagation; identity when
        ``None``.
    rotation : LunarRotationDynamics or None, optional
        Integrated lunar rotation; orbit only when ``None``.
    kuiper_belt_ring : KuiperBeltRing or None, optional
        Prepropagated Kuiper-belt ring whose points act as sources.
    solar_gm_variation : SolarGmVariation or None, optional
        Linear drift of the solar GM.

    Attributes
    ----------
    body_names : tuple of str
        All bodies in kernel order: Earth, Moon, external bodies, ring
        points.
    gravitational_parameters_m3_s2 : numpy.ndarray
        Nominal GM of every body in m³ s⁻², shape ``(N,)``.
    prescribed_body_names : tuple of str
        Bodies read from the ephemeris: the EMB and the external bodies.

    Raises
    ------
    ValueError
        If the force and torque models cannot be compiled into the native
        kernel, for example when a required body such as the Sun is missing.
    """

    def __init__(
        self,
        earth_body: GravitatingBody,
        moon_body: GravitatingBody,
        perturbing_bodies: Sequence[GravitatingBody],
        ephemeris: Ephemeris,
        force_group: LunarForceGroup | None = None,
        *,
        earth_fixed2inertial_matrix_provider: OrientationProvider | None = None,
        moon_fixed2inertial_matrix_provider: OrientationProvider | EphemerisLunarAxes | None = None,
        rotation: LunarRotationDynamics | None = None,
        kuiper_belt_ring: KuiperBeltRing | None = None,
        solar_gm_variation: SolarGmVariation | None = None,
    ) -> None:
        self.earth_body = earth_body
        self.moon_body = moon_body
        self.perturbing_bodies = tuple(perturbing_bodies)
        self.ephemeris = ephemeris
        self.force_group = LunarForceGroup() if force_group is None else force_group
        self.rotation = rotation
        self.kuiper_belt_ring = kuiper_belt_ring
        self.solar_gm_variation = solar_gm_variation
        bodies = (
            earth_body,
            moon_body,
            *self.perturbing_bodies,
            *(() if kuiper_belt_ring is None else kuiper_belt_ring.bodies),
        )
        self.body_names = tuple(body.body_id for body in bodies)
        self.gravitational_parameters_m3_s2 = np.asarray(
            [body.gravitational_parameter_m3_s2 for body in bodies], dtype=float
        )
        self.prescribed_body_names = (EMB, *(body.body_id for body in self.perturbing_bodies))
        self._earth_axes = earth_fixed2inertial_matrix_provider
        self._moon_axes = moon_fixed2inertial_matrix_provider
        self._fused = compile_fused_rhs(self)

    @property
    def epoch_inputs(self) -> EpochInputs:
        """Picklable recipe of the time-only inputs, reflecting the current ring and GM drift."""
        return EpochInputs(
            body_names=self.body_names,
            gravitational_parameters_m3_s2=self.gravitational_parameters_m3_s2,
            prescribed_body_names=self.prescribed_body_names,
            solar_gm_variation=self.solar_gm_variation,
            kuiper_belt_ring=self.kuiper_belt_ring,
            earth_axes=self._earth_axes,
            moon_axes=self._moon_axes,
            point_mass=self.force_group.has_point_mass,
            earth_c20=self._fused.earth_c20,
            earth_j2=self._fused.earth_j2,
        )

    def epoch_batch(self, epochs_tdb: Sequence[Epoch]) -> EpochBatch:
        """Build the time-only inputs at a batch of epochs.

        Parameters
        ----------
        epochs_tdb : sequence of Epoch
            TDB epochs.

        Returns
        -------
        EpochBatch
            Stacked prescribed inputs.
        """
        return self.epoch_inputs.build(self.ephemeris, epochs_tdb)

    def terms(self, epoch_tdb, vector, batch=None, *, index=0, history=None):
        """Return the derivative together with every force and torque term.

        Parameters
        ----------
        epoch_tdb : Epoch
            TDB epoch.
        vector : array_like
            Integration state, shape ``(d,)``.
        batch : EpochBatch or None, optional
            Prebuilt inputs containing ``epoch_tdb``; built when ``None``.
        index : int, default 0
            Row of ``epoch_tdb`` in ``batch``.
        history : StateHistoryProvider or None, optional
            History for delayed tides and deformation.

        Returns
        -------
        dict
            ``"forces"``: Moon accelerations in m s⁻² per active force name,
            each of shape ``(3,)`` with ICRF axes; ``"earth_forces"``: the
            Earth accelerations of the same terms, so that the relative
            acceleration is their difference; ``"torques"``: per-model
            torques in PA axes, s⁻², shape ``(n_torques, 3)``;
            ``"rotation"``: viscous and pressure core-mantle torques and the
            geodetic precession rate, shape ``(3, 3)``; ``"inertia"``: total
            inertia, mantle inertia and inertia rate, shape ``(3, 3, 3)``;
            ``"derivative"``: the state derivative, shape ``(d,)``.

        Raises
        ------
        ValueError
            If the state is degenerate (zero quaternion, coincident bodies,
            singular mantle inertia) or a delayed epoch lies outside the
            history.
        """
        batch = self.epoch_batch((epoch_tdb,)) if batch is None else batch
        return self._fused.terms(batch, index, self.prepare_step(epoch_tdb, batch, history, index=index), vector)

    def prepare_step(
        self, epoch: Epoch, batch: EpochBatch, history, *, index: int = 0, rigid: bool = False
    ) -> StepData:
        """Evaluate the candidate-independent data of one integration step.

        Parameters
        ----------
        epoch : Epoch
            TDB epoch of the step.
        batch : EpochBatch
            Prescribed inputs containing ``epoch``.
        history : StateHistoryProvider or None
            Accepted history for delayed tides and deformation.
        index : int, default 0
            Row of ``epoch`` in ``batch``.
        rigid : bool, default False
            Use the undeformed inertia, as on the rigid seed arc.

        Returns
        -------
        StepData
            Force payloads and the delayed lunar inertia.

        Raises
        ------
        ValueError
            If a delayed epoch lies outside the history.
        """
        inertia = None
        if self.rotation is not None and self.rotation.deformation is not None:
            inertia = self.rotation.deformation.evaluate(epoch, history, rigid=rigid)
        return StepData(self._fused.prepare_step(epoch, batch, index, history), inertia)

    def derivatives(
        self,
        epoch_tdb: Epoch,
        vector,
        batch: EpochBatch | None = None,
        *,
        history=None,
        index: int = 0,
        rigid: bool = False,
        step: StepData | None = None,
    ) -> np.ndarray:
        """Return the time derivative of the integration state.

        Parameters
        ----------
        epoch_tdb : Epoch
            TDB epoch.
        vector : array_like
            Integration state, shape ``(d,)``.
        batch : EpochBatch or None, optional
            Prescribed inputs containing ``epoch_tdb``; built when ``None``.
        history : StateHistoryProvider or None, optional
            History for delayed tides and deformation.
        index : int, default 0
            Row of ``epoch_tdb`` in ``batch``.
        rigid : bool, default False
            Use the constant undeformed inertia of the rigid seed model.
        step : StepData or None, optional
            Result of :meth:`prepare_step` for this epoch; computed when
            ``None``.

        Returns
        -------
        numpy.ndarray
            State derivative of shape ``(d,)``: velocity (m s⁻¹), relative
            acceleration (m s⁻²), quaternion rate (s⁻¹) and angular
            accelerations (rad s⁻²).

        Raises
        ------
        ValueError
            If the state is degenerate (zero quaternion, coincident bodies,
            singular mantle inertia) or a delayed epoch lies outside the
            history.
        """
        y = np.asarray(vector, dtype=float)
        batch = self.epoch_batch((epoch_tdb,)) if batch is None else batch
        if step is None:
            step = self.prepare_step(epoch_tdb, batch, history, index=index, rigid=rigid)
        return self._fused.derivative(batch, index, step, y)

    def split_derivatives(
        self,
        epoch_tdb: Epoch,
        vector,
        vector_low,
        batch: EpochBatch | None = None,
        *,
        history=None,
        index: int = 0,
        step: StepData | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the time derivative of a double-double state.

        The arguments are those of :meth:`derivatives` with the low part
        ``vector_low`` of the state added.

        Returns
        -------
        derivative, derivative_low : numpy.ndarray
            High and low parts of the state derivative, shape ``(d,)``.

        Raises
        ------
        ValueError
            As for :meth:`derivatives`.
        """
        batch = self.epoch_batch((epoch_tdb,)) if batch is None else batch
        if step is None:
            step = self.prepare_step(epoch_tdb, batch, history, index=index)
        return self._fused.split_derivative(batch, index, step, vector, vector_low)
