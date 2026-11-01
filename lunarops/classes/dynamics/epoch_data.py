"""Time-only dynamics inputs never depend on the integrated Moon state.

They include prescribed states, GM drift, ring nodes, Earth/Moon axes,
Newtonian source terms and delayed prescribed tide states."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from lunarops._dynamics_core import point_mass_symmetric_batch
from lunarops.classes.time import Epoch

from .earth_figure_orientation import earth_j2_increment
from .solar_environment import KuiperBeltRing, SolarGmVariation

_IDENTITY3 = np.eye(3)
_IDENTITY3.setflags(write=False)


@dataclass(frozen=True, slots=True)
class EpochBatch:
    """Stacked prescribed inputs; Earth/Moon axes rotate fixed coordinates to inertial.

    The candidate Moon row is filled only in the kernel workspace.
    """

    positions_m: np.ndarray
    velocities_mps: np.ndarray
    gravitational_parameters_m3_s2: np.ndarray
    earth_fixed2inertial_matrix: np.ndarray
    moon_fixed2inertial_matrix: np.ndarray
    ephemeris_earth_acceleration_mps2: np.ndarray
    source_accelerations_mps2: np.ndarray
    source_potentials_m2_s2: np.ndarray
    earth_c20: np.ndarray
    earth_j2: np.ndarray


@dataclass(frozen=True, slots=True)
class OrientationProvider:
    """Fixed-to-inertial matrices of one body."""

    matrix: Callable[[Epoch], np.ndarray]


    def matrices(self, epochs: Sequence[Epoch], ephemeris=None) -> np.ndarray:
        return np.asarray([self.matrix(epoch) for epoch in epochs], dtype=float)


@dataclass(frozen=True, slots=True)
class EphemerisLunarAxes:
    """Lunar principal axes from whichever ephemeris handle the calling process owns."""

    def matrices(self, epochs: Sequence[Epoch], ephemeris) -> np.ndarray:
        return np.asarray(ephemeris.lunar_orientation.pa_to_lcrs_matrices(tuple(epochs)), dtype=float)


@dataclass(frozen=True, slots=True)
class EpochInputs:
    """Prescribed inputs in body order: Earth, Moon, external bodies, ring nodes."""

    body_names: tuple[str, ...]
    gravitational_parameters_m3_s2: np.ndarray
    prescribed_body_names: tuple[str, ...]
    solar_gm_variation: SolarGmVariation | None
    kuiper_belt_ring: KuiperBeltRing | None
    earth_axes: OrientationProvider | EphemerisLunarAxes | None
    moon_axes: OrientationProvider | EphemerisLunarAxes | None
    point_mass: bool
    earth_c20: tuple[float, float, float, float]
    earth_j2: tuple[float, float, float, float]

    def _axes(self, provider, epochs, ephemeris) -> np.ndarray:
        if provider is None:
            return np.broadcast_to(_IDENTITY3, (len(epochs), 3, 3))
        return provider.matrices(epochs, ephemeris)

    def build(self, ephemeris, epochs: Sequence[Epoch]) -> EpochBatch:
        epochs = tuple(epochs)
        count = len(epochs)
        prescribed = len(self.prescribed_body_names)
        states = ephemeris.body_state_matrices_bcrs(self.prescribed_body_names, epochs)
        positions = np.zeros((count, len(self.body_names), 3), dtype=float)
        velocities = np.zeros_like(positions)
        positions[:, 0] = states[:, 0, :3]
        velocities[:, 0] = states[:, 0, 3:]
        positions[:, 2 : prescribed + 1] = states[:, 1:, :3]
        velocities[:, 2 : prescribed + 1] = states[:, 1:, 3:]
        gm = np.empty((count, len(self.body_names)), dtype=float)
        gm[:] = self.gravitational_parameters_m3_s2
        if self.solar_gm_variation is not None:
            sun = self.body_names.index("SUN")
            for index, epoch in enumerate(epochs):
                gm[index, sun] *= self.solar_gm_variation.factor(epoch)
        if self.kuiper_belt_ring is not None:
            sun = self.body_names.index("SUN")
            positions[:, prescribed + 1 :], velocities[:, prescribed + 1 :] = self.kuiper_belt_ring.states_many(
                epochs, positions[:, sun], velocities[:, sun]
            )
        source_acceleration = np.zeros((count, len(self.body_names) - 2, 3))
        source_potential = np.zeros((count, len(self.body_names) - 2))
        if self.point_mass:
            source_acceleration, source_potential = point_mass_symmetric_batch(
                np.ascontiguousarray(positions[:, 2:]), np.ascontiguousarray(gm[:, 2:])
            )
        c20, reference, rate, quadratic = self.earth_c20
        earth_c20 = np.asarray([c20 - earth_j2_increment(epoch, reference, rate, quadratic) / np.sqrt(5.0)
                                for epoch in epochs])
        j2, reference, rate, quadratic = self.earth_j2
        earth_j2 = np.asarray([j2 + earth_j2_increment(epoch, reference, rate, quadratic) for epoch in epochs])
        return EpochBatch(
            positions, velocities, gm, self._axes(self.earth_axes, epochs, ephemeris),
            self._axes(self.moon_axes, epochs, ephemeris), ephemeris.body_accelerations_bcrs("EARTH", epochs),
            source_acceleration, source_potential, earth_c20, earth_j2,
        )


class SerialEpochSource:

    def __init__(self, inputs: EpochInputs, ephemeris) -> None:
        self.inputs = inputs
        self.ephemeris = ephemeris

    def stream(self, initial_epoch: Epoch, offset_batches: Sequence[Sequence[float]]) -> Iterator[EpochBatch]:
        for offsets in offset_batches:
            yield self.inputs.build(self.ephemeris, batch_epochs(initial_epoch, offsets))


def batch_epochs(initial_epoch: Epoch, offsets: Sequence[float]) -> tuple[Epoch, ...]:
    return tuple(initial_epoch.shifted(float(offset)) for offset in offsets)
