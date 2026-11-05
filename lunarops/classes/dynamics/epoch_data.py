"""Time-only inputs of the lunar right-hand side.

Everything here depends on the epoch alone, never on the integrated Moon
state: prescribed BCRS body states, the solar GM drift, Kuiper-belt ring
nodes, Earth and Moon body axes, the Newtonian source/source terms and the
time-variable Earth zonal coefficients. The inputs are built in batches of
epochs so that ephemeris queries are vectorised.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from lunarops._dynamics_core import point_mass_symmetric_batch
from lunarops.classes.time import Epoch

from .earth_figure_orientation import earth_j2_increment
from .solar_environment import KuiperBeltRing, SolarGmVariation

_IDENTITY3 = np.eye(3)


@dataclass(frozen=True, slots=True)
class EpochBatch:
    """Prescribed inputs of a batch of ``E`` epochs for ``N`` bodies.

    Body order is Earth, Moon, external bodies, Kuiper-belt ring nodes. The
    Moon row (index 1) of the position and velocity arrays is zero: the
    native kernel fills it from the candidate state.

    Attributes
    ----------
    positions_m : numpy.ndarray
        BCRS positions with ICRF axes in m, shape ``(E, N, 3)``.
    velocities_mps : numpy.ndarray
        BCRS velocities with ICRF axes in m s⁻¹, shape ``(E, N, 3)``.
    gravitational_parameters_m3_s2 : numpy.ndarray
        GM of every body in m³ s⁻², including the solar GM drift,
        shape ``(E, N)``.
    earth_fixed2inertial_matrix : numpy.ndarray
        Rotation from Earth gravity-field axes to ICRF axes, shape
        ``(E, 3, 3)``; identity when no Earth axes are configured.
    moon_fixed2inertial_matrix : numpy.ndarray
        Rotation from Moon PA axes to ICRF axes taken from the ephemeris
        lunar orientation, shape ``(E, 3, 3)``; identity when not
        configured (the integrated attitude is used instead).
    ephemeris_earth_acceleration_mps2 : numpy.ndarray
        Ephemeris BCRS acceleration of the Earth in m s⁻², shape ``(E, 3)``.
    source_accelerations_mps2 : numpy.ndarray
        Newtonian accelerations of every external source (bodies 2 to
        ``N - 1``) due to the other external sources, in m s⁻²,
        shape ``(E, N - 2, 3)``; zeros when the point-mass force is off.
    source_potentials_m2_s2 : numpy.ndarray
        Matching Newtonian potentials ``Σ GM_j / r_ij`` in m² s⁻²,
        shape ``(E, N - 2)``.
    earth_c20 : numpy.ndarray
        Fully normalised Earth C̄20 including the J2 time variation,
        shape ``(E,)``.
    earth_j2 : numpy.ndarray
        Unnormalised Earth J2 including its time variation, used by the
        Earth figure torque, shape ``(E,)``.
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
    """Body-fixed-to-inertial rotation of one body from a scalar function.

    Attributes
    ----------
    matrix : Callable[[Epoch], numpy.ndarray]
        Function of a TDB epoch returning the ``(3, 3)`` rotation from
        body-fixed to ICRF axes, for example
        :func:`~lunarops.classes.dynamics.earth_figure_orientation.de440_earth_fixed2inertial_matrix`.
    """

    matrix: Callable[[Epoch], np.ndarray]

    def matrices(self, epochs: Sequence[Epoch], ephemeris=None) -> np.ndarray:
        """Return the rotation matrices at several epochs.

        Parameters
        ----------
        epochs : sequence of Epoch
            TDB epochs.
        ephemeris : object, optional
            Ignored; present for interface compatibility with
            :class:`EphemerisLunarAxes`.

        Returns
        -------
        numpy.ndarray
            Body-fixed-to-ICRF matrices of shape ``(len(epochs), 3, 3)``.
        """
        return np.asarray([self.matrix(epoch) for epoch in epochs], dtype=float)


@dataclass(frozen=True, slots=True)
class EphemerisLunarAxes:
    """Lunar PA-to-inertial rotation read from the ephemeris lunar orientation.

    The record holds no ephemeris handle, so it pickles cleanly; each process
    passes its own ephemeris to :meth:`matrices`.
    """

    def matrices(self, epochs: Sequence[Epoch], ephemeris) -> np.ndarray:
        """Return the lunar PA-to-LCRS matrices at several epochs.

        Parameters
        ----------
        epochs : sequence of Epoch
            TDB epochs.
        ephemeris : Ephemeris
            Ephemeris providing ``lunar_orientation``.

        Returns
        -------
        numpy.ndarray
            Rotation matrices from Moon PA axes to LCRS (ICRF-aligned) axes,
            shape ``(len(epochs), 3, 3)``.
        """
        return np.asarray(ephemeris.lunar_orientation.pa_to_lcrs_matrices(tuple(epochs)), dtype=float)


@dataclass(frozen=True, slots=True)
class EpochInputs:
    """Picklable recipe for building :class:`EpochBatch` records.

    Attributes
    ----------
    body_names : tuple of str
        All bodies in kernel order: Earth, Moon, external bodies, ring nodes.
    gravitational_parameters_m3_s2 : numpy.ndarray
        Nominal GM of every body in m³ s⁻², shape ``(N,)``.
    prescribed_body_names : tuple of str
        Bodies read from the ephemeris: Earth followed by the external bodies.
    solar_gm_variation : SolarGmVariation or None
        Linear drift of the solar GM; ``None`` for a constant GM.
    kuiper_belt_ring : KuiperBeltRing or None
        Prepropagated Kuiper-belt ring; ``None`` without a ring.
    earth_axes : OrientationProvider, EphemerisLunarAxes or None
        Earth gravity-field axes; ``None`` gives identity matrices.
    moon_axes : OrientationProvider, EphemerisLunarAxes or None
        Prescribed lunar PA axes; ``None`` gives identity matrices.
    point_mass : bool
        Whether to precompute the Newtonian source/source terms.
    earth_c20 : tuple of float
        ``(C̄20, reference_jd_tdb, linear_per_year, quadratic_per_year2)`` of
        the fully normalised Earth C̄20; the J2 polynomial increment is
        divided by √5 and subtracted.
    earth_j2 : tuple of float
        ``(J2, reference_jd_tdb, linear_per_year, quadratic_per_year2)`` of
        the unnormalised Earth J2 used by the Earth figure torque.
    """

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
        """Evaluate the prescribed inputs at a batch of epochs.

        Parameters
        ----------
        ephemeris : Ephemeris
            Ephemeris providing BCRS states, accelerations and, when
            configured, the lunar orientation.
        epochs : sequence of Epoch
            TDB epochs of the batch.

        Returns
        -------
        EpochBatch
            Stacked inputs with leading dimension ``len(epochs)``.

        Raises
        ------
        ValueError
            If the solar GM becomes non-positive, a ring epoch lies outside
            its prepropagated interval, or two source bodies coincide.
        """
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
    """In-process producer of epoch batches for the integrator's main grid.

    Parameters
    ----------
    inputs : EpochInputs
        Recipe of the time-only inputs.
    ephemeris : Ephemeris
        Ephemeris used to build every batch.
    """

    def __init__(self, inputs: EpochInputs, ephemeris) -> None:
        self.inputs = inputs
        self.ephemeris = ephemeris

    def stream(self, initial_epoch: Epoch, offset_batches: Sequence[Sequence[float]]) -> Iterator[EpochBatch]:
        """Yield one epoch batch per group of offsets.

        Parameters
        ----------
        initial_epoch : Epoch
            TDB reference epoch of the offsets.
        offset_batches : sequence of sequence of float
            Groups of time offsets from ``initial_epoch`` in TDB seconds.

        Yields
        ------
        EpochBatch
            Inputs at the epochs of each group, in order.
        """
        for offsets in offset_batches:
            yield self.inputs.build(self.ephemeris, batch_epochs(initial_epoch, offsets))


def batch_epochs(initial_epoch: Epoch, offsets: Sequence[float]) -> tuple[Epoch, ...]:
    """Return the epochs at given offsets from a reference epoch.

    Parameters
    ----------
    initial_epoch : Epoch
        Reference epoch.
    offsets : sequence of float
        Offsets in seconds of the reference epoch's time scale.

    Returns
    -------
    tuple of Epoch
        ``initial_epoch`` shifted by each offset.
    """
    return tuple(initial_epoch.shifted(float(offset)) for offset in offsets)
