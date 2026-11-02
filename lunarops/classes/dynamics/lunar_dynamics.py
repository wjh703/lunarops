"""Moon-relative translation and optional mantle/core rotational dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.time import Epoch

from .bodies import GravitatingBody
from .context import StepData
from .epoch_data import EphemerisLunarAxes, EpochBatch, EpochInputs, OrientationProvider
from .force_models import LunarForceGroup
from .fused_rhs import compile_fused_rhs
from .rotation import LunarRotationDynamics
from .solar_environment import KuiperBeltRing, SolarGmVariation


class LunarDynamics:

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
        self.prescribed_body_names = ("EARTH", *(body.body_id for body in self.perturbing_bodies))
        # Prescribed bodies the Earth tide reads at its fixed delays; they depend on time
        # only, so they are fetched together with the rest of the epoch data.
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
        return self.epoch_inputs.build(self.ephemeris, epochs_tdb)

    def terms(self, epoch_tdb, vector, batch=None, *, index=0, history=None):
        batch = self.epoch_batch((epoch_tdb,)) if batch is None else batch
        return self._fused.terms(batch, index, self.prepare_step(epoch_tdb, batch, history, index=index), vector)


    def prepare_step(self, epoch: Epoch, batch: EpochBatch, history, *, index: int = 0, rigid: bool = False) -> StepData:
        """Evaluate everything that depends on accepted history but not on the candidate state."""
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
        """Return the state derivative; ``rigid`` uses the constant-inertia backward seed model.

        ``step`` carries the candidate-independent data of :meth:`prepare_step`.
        """
        y = np.asarray(vector, dtype=float)
        batch = self.epoch_batch((epoch_tdb,)) if batch is None else batch
        if step is None:
            step = self.prepare_step(epoch_tdb, batch, history, index=index, rigid=rigid)
        return self._fused.derivative(batch, index, step, y)
