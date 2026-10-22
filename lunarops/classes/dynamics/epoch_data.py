"""Time-only dynamics inputs, built in-process or by MPI workers.

Everything here depends on the epoch alone, never on the integrated Moon:
prescribed body states, GM drift, the Kuiper ring, Earth/Moon axes, the
Newtonian source/source cache and the tide's delayed prescribed states.
:class:`EpochInputs` is picklable, so rank 0 and every MPI worker run the same
functions on the same epochs and obtain bit-identical arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from lunarops._dynamics_core import point_mass_symmetric_batch
from lunarops.classes.time import Epoch

from .forces import PointMassGravityCache
from .solar_environment import KuiperBeltRing, SolarGmVariation

EpochArrays = dict[str, np.ndarray | None]

_IDENTITY3 = np.eye(3)
_IDENTITY3.setflags(write=False)


@dataclass(frozen=True, slots=True)
class DynamicsEpochData:
    """Prescribed inputs of one TDB epoch; the Moon row is filled from the candidate state.

    ``earth_fixed2inertial_matrix`` rotates Earth gravity axes to inertial axes;
    ``moon_fixed2inertial_matrix`` rotates lunar principal axes to inertial axes.
    ``delayed_external_states`` holds the tide's prescribed bodies at its fixed delays,
    shaped ``(delay, body, 6)``.
    """

    epoch_tdb: Epoch
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    earth_fixed2inertial_matrix: np.ndarray
    moon_fixed2inertial_matrix: np.ndarray
    ephemeris_earth_acceleration_mps2: np.ndarray
    gravitational_parameters_m3_s2: np.ndarray
    point_mass_gravity_cache: PointMassGravityCache | None = None
    delayed_external_states: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class OrientationProvider:
    """Fixed-to-inertial matrices of one body; picklable when ``matrix`` is a module-level function."""

    matrix: Callable[[Epoch], np.ndarray]

    @classmethod
    def from_scalar(cls, matrix: Callable[[Epoch], np.ndarray]) -> OrientationProvider:
        return cls(matrix)

    def matrices(self, epochs: Sequence[Epoch], ephemeris=None) -> np.ndarray:
        return np.asarray([self.matrix(epoch) for epoch in epochs], dtype=float)


@dataclass(frozen=True, slots=True)
class EphemerisLunarAxes:
    """Lunar principal axes from whichever ephemeris handle the calling process owns."""

    def matrices(self, epochs: Sequence[Epoch], ephemeris) -> np.ndarray:
        return np.asarray(ephemeris.lunar_orientation.pa_to_lcrs_matrices(tuple(epochs)), dtype=float)


@dataclass(frozen=True, slots=True)
class EpochInputs:
    """Picklable recipe of the time-only inputs; body order is Earth, Moon, prescribed, ring."""

    body_names: tuple[str, ...]
    gravitational_parameters_m3_s2: np.ndarray
    prescribed_body_names: tuple[str, ...]
    solar_gm_variation: SolarGmVariation | None
    kuiper_belt_ring: KuiperBeltRing | None
    earth_axes: OrientationProvider | EphemerisLunarAxes | None
    moon_axes: OrientationProvider | EphemerisLunarAxes | None
    point_mass: bool
    delayed_external_names: tuple[str, ...]
    delayed_offsets_s: tuple[float, ...]

    def _axes(self, provider, epochs, ephemeris) -> np.ndarray:
        if provider is None:
            return np.broadcast_to(_IDENTITY3, (len(epochs), 3, 3))
        return provider.matrices(epochs, ephemeris)

    def arrays(self, ephemeris, epochs: Sequence[Epoch]) -> EpochArrays:
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
        arrays: EpochArrays = {
            "positions": positions,
            "velocities": velocities,
            "gm": gm,
            "earthAcceleration": ephemeris.body_accelerations_bcrs("EARTH", epochs),
            "earthAxes": self._axes(self.earth_axes, epochs, ephemeris),
            "moonAxes": self._axes(self.moon_axes, epochs, ephemeris),
            "sourceAccelerations": None,
            "sourcePotentials": None,
            "delayedStates": None,
        }
        if self.point_mass:
            arrays["sourceAccelerations"], arrays["sourcePotentials"] = point_mass_symmetric_batch(
                np.ascontiguousarray(positions[:, 2:]), np.ascontiguousarray(gm[:, 2:])
            )
        if self.delayed_external_names:
            delayed = tuple(epoch.shifted(offset) for epoch in epochs for offset in self.delayed_offsets_s)
            arrays["delayedStates"] = ephemeris.body_state_matrices_bcrs(self.delayed_external_names, delayed).reshape(
                count, len(self.delayed_offsets_s), len(self.delayed_external_names), 6
            )
        return arrays

    @staticmethod
    def records(epochs: Sequence[Epoch], arrays: EpochArrays) -> tuple[DynamicsEpochData, ...]:
        for value in arrays.values():
            if value is not None and value.flags.writeable:
                value.setflags(write=False)
        accelerations, potentials, delayed = (
            arrays["sourceAccelerations"],
            arrays["sourcePotentials"],
            arrays["delayedStates"],
        )
        positions, velocities, gm = arrays["positions"], arrays["velocities"], arrays["gm"]
        earth_axes, moon_axes, earth_acceleration = arrays["earthAxes"], arrays["moonAxes"], arrays["earthAcceleration"]
        assert positions is not None and velocities is not None and gm is not None
        assert earth_axes is not None and moon_axes is not None and earth_acceleration is not None
        return tuple(
            DynamicsEpochData(
                epoch_tdb=epoch,
                positions_m=positions[index],
                velocities_mps=velocities[index],
                earth_fixed2inertial_matrix=earth_axes[index],
                moon_fixed2inertial_matrix=moon_axes[index],
                ephemeris_earth_acceleration_mps2=earth_acceleration[index],
                gravitational_parameters_m3_s2=gm[index],
                point_mass_gravity_cache=None
                if accelerations is None or potentials is None
                else PointMassGravityCache(2, accelerations[index], potentials[index]),
                delayed_external_states=None if delayed is None else delayed[index],
            )
            for index, epoch in enumerate(epochs)
        )


class SerialEpochSource:
    """Build epoch inputs in-process from the run's own ephemeris."""

    def __init__(self, inputs: EpochInputs, ephemeris) -> None:
        self.inputs = inputs
        self.ephemeris = ephemeris

    def stream(self, initial_epoch: Epoch, offset_batches: Sequence[Sequence[float]]) -> Iterator[EpochArrays]:
        for offsets in offset_batches:
            yield self.inputs.arrays(self.ephemeris, batch_epochs(initial_epoch, offsets))


def batch_epochs(initial_epoch: Epoch, offsets: Sequence[float]) -> tuple[Epoch, ...]:
    """The one expression for grid epochs, shared by rank 0 and MPI workers."""
    return tuple(initial_epoch.shifted(float(offset)) for offset in offsets)
