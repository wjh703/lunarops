"""Solar-system environment terms: solar GM drift and the Kuiper-belt ring.

:class:`SolarGmVariation` scales the solar GM linearly in time.
:class:`KuiperBeltRing` represents the Kuiper belt as equal point masses on
a ring whose trajectories are prepropagated once and then act as prescribed
gravitating sources.

Attributes
----------
AU_M : float
    Astronomical unit in m (IAU 2012).
JULIAN_YEAR_DAYS : float
    Julian year in days.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from lunarops.classes.time import Epoch, TimeScale

from .bodies import GravitatingBody

AU_M = 149_597_870_700.0
JULIAN_YEAR_DAYS = 365.25


@dataclass(frozen=True, slots=True)
class SolarGmVariation:
    """Linear drift of the solar gravitational parameter.

    ``GM_sun(t) = GM_sun · (1 + rate · t)`` with ``t`` in Julian years of TDB
    since ``reference_jd_tdb``.

    Attributes
    ----------
    fractional_rate_per_year : float
        Fractional rate ``d(GM)/dt / GM`` per Julian year.
    reference_jd_tdb : float
        Reference epoch as a TDB Julian date.
    """

    fractional_rate_per_year: float
    reference_jd_tdb: float

    def factor(self, epoch: Epoch) -> float:
        """Return the GM scale factor at one epoch.

        Parameters
        ----------
        epoch : Epoch
            Evaluation epoch; its two-part Julian date is read as TDB.

        Returns
        -------
        float
            ``1 + rate · t``.

        Raises
        ------
        ValueError
            If the factor is not positive.
        """
        years = ((epoch.jd1 - self.reference_jd_tdb) + epoch.jd2) / JULIAN_YEAR_DAYS
        factor = 1.0 + self.fractional_rate_per_year * years
        if factor <= 0:
            raise ValueError("Solar GM must remain positive over the propagation interval")
        return factor


@dataclass(frozen=True, slots=True)
class KuiperBeltRing:
    """Kuiper belt modelled as a ring of equal point masses.

    At ``reference_jd_tdb`` the points lie equally spaced on a circle of
    radius ``radius_m`` around the Sun in the J2000 ecliptic plane, moving on
    circular orbits. :meth:`prepare` then propagates every point in the BCRS
    under the prescribed sources, without point-to-point attraction
    (Pitjeva & Pitjev 2018). Radius, point mass and count are configured to
    the DE440 ring values (Park et al. 2021); the initial phase is an
    explicit parameter because the DE440 SPK file does not carry the
    ring-point initial conditions.

    Attributes
    ----------
    point_gm_m3_s2 : float
        GM of each ring point in m³ s⁻².
    solar_gm_m3_s2 : float
        Solar GM in m³ s⁻² defining the circular angular rate.
    radius_m : float
        Ring radius in m.
    point_count : int
        Number of ring points.
    obliquity_rad : float
        Obliquity of the J2000 ecliptic relative to the ICRF equator in rad.
    reference_jd_tdb : float
        Epoch of the circular initial configuration as a TDB Julian date.
    phase_rad : float, default 0.0
        Ecliptic longitude of the first point at ``reference_jd_tdb`` in rad.

    References
    ----------
    Park et al. (2021); Pitjeva & Pitjev (2018).
    """

    point_gm_m3_s2: float
    solar_gm_m3_s2: float
    radius_m: float
    point_count: int
    obliquity_rad: float
    reference_jd_tdb: float
    phase_rad: float = 0.0
    _bodies: tuple = field(init=False, repr=False, compare=False)
    _solution: Any = field(default=None, repr=False, compare=False)
    _backward_solution: Any = field(default=None, repr=False, compare=False)
    _initial_states: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Create the ring-point bodies ``KBO_RING_01``, ``KBO_RING_02``, ..."""
        object.__setattr__(
            self,
            "_bodies",
            tuple(GravitatingBody(f"KBO_RING_{i + 1:02d}", self.point_gm_m3_s2) for i in range(self.point_count)),
        )

    @property
    def bodies(self):
        """Ring points as :class:`GravitatingBody` records, in ring order."""
        return self._bodies

    def circular_states(self, epoch: Epoch, sun_position, sun_velocity):
        """Return the ring-point states on unperturbed circular orbits.

        Parameters
        ----------
        epoch : Epoch
            Evaluation epoch; its two-part Julian date is read as TDB.
        sun_position : array_like
            BCRS position of the Sun in m, shape ``(3,)``.
        sun_velocity : array_like
            BCRS velocity of the Sun in m s⁻¹, shape ``(3,)``.

        Returns
        -------
        position : numpy.ndarray
            BCRS positions with ICRF axes in m, shape ``(point_count, 3)``.
        velocity : numpy.ndarray
            BCRS velocities with ICRF axes in m s⁻¹, shape ``(point_count, 3)``.
        """
        omega = np.sqrt(self.solar_gm_m3_s2 / self.radius_m**3)
        elapsed = ((epoch.jd1 - self.reference_jd_tdb) + epoch.jd2) * 86400.0
        angles = self.phase_rad + np.arange(self.point_count) * (2 * np.pi / self.point_count) + omega * elapsed
        c, s = np.cos(angles), np.sin(angles)
        ce, se = np.cos(self.obliquity_rad), np.sin(self.obliquity_rad)
        position = self.radius_m * np.column_stack((c, ce * s, se * s)) + sun_position
        velocity = self.radius_m * omega * np.column_stack((-s, ce * c, se * c)) + sun_velocity
        return position, velocity

    def prepare(self, ephemeris, source_names, source_gm, initial_epoch, duration_s, solar_gm_variation):
        """Return a copy of the ring with prepropagated point trajectories.

        The circular configuration at ``reference_jd_tdb`` is propagated with
        DOP853 (dense output) forward and backward as needed to cover
        ``[initial_epoch, initial_epoch + duration_s]``. With a solar GM
        drift, the initial heliocentric velocities are scaled by the square
        root of the GM factor at the reference epoch.

        Parameters
        ----------
        ephemeris : Ephemeris
            Ephemeris providing BCRS source states.
        source_names : tuple of str
            Attracting bodies; must include ``"SUN"``.
        source_gm : array_like
            GM of each source in m³ s⁻², shape ``(len(source_names),)``.
        initial_epoch : Epoch
            Start of the interval to cover.
        duration_s : float
            Length of the interval in TDB seconds.
        solar_gm_variation : SolarGmVariation or None
            Solar GM drift applied to the Sun.

        Returns
        -------
        KuiperBeltRing
            New ring holding the dense-output solutions.

        Raises
        ------
        ImportError
            If SciPy is not installed.
        RuntimeError
            If the propagation fails.
        """
        reference = Epoch(self.reference_jd_tdb, 0.0, TimeScale.TDB)
        sun = source_names.index("SUN")
        sun_state = ephemeris.body_state_matrix_bcrs(("SUN",), reference)[0]
        position, velocity = self.circular_states(reference, sun_state[:3], sun_state[3:])
        if solar_gm_variation is not None:
            velocity = sun_state[3:] + (velocity - sun_state[3:]) * np.sqrt(solar_gm_variation.factor(reference))
        solutions: dict[str, Any] = {"_initial_states": (position, velocity)}
        initial = np.concatenate((position.ravel(), velocity.ravel()))
        source_gm = np.asarray(source_gm, dtype=float)

        def rhs(seconds, state):
            epoch = reference.shifted(float(seconds))
            sources = ephemeris.body_state_matrix_bcrs(source_names, epoch)[:, :3]
            gm = source_gm.copy()
            if solar_gm_variation is not None:
                gm[sun] *= solar_gm_variation.factor(epoch)
            points = state[: self.point_count * 3].reshape(-1, 3)
            delta = sources[None, :, :] - points[:, None, :]
            distances = np.linalg.norm(delta, axis=2)
            acceleration = np.sum(delta * (gm[None, :] / distances**3)[:, :, None], axis=1)
            return np.concatenate((state[self.point_count * 3 :], acceleration.ravel()))

        offsets = (reference.seconds_until(initial_epoch), reference.seconds_until(initial_epoch) + duration_s)
        # Cover the requested interval on both sides of the independent ring
        # initial epoch. No integration step is taken beyond these endpoints.
        for final, attribute in ((max(0, *offsets), "_solution"), (min(0, *offsets), "_backward_solution")):
            if final == 0:
                continue
            try:
                from scipy.integrate import solve_ivp
            except ImportError as exc:
                raise ImportError("Kuiper belt propagation requires SciPy; install lunarops[dynamics].") from exc
            solution = solve_ivp(
                rhs,
                (0, final),
                initial,
                method="DOP853",
                dense_output=True,
                rtol=3e-14,
                atol=np.r_[np.full(self.point_count * 3, 0.01), np.full(self.point_count * 3, 1e-8)],
                max_step=14 * 86400.0,
            )
            if not solution.success:
                raise RuntimeError(f"KBO ring prepropagation failed: {solution.message}")
            solutions[attribute] = solution.sol
        return replace(self, **solutions)

    def states_many(self, epochs, sun_positions, sun_velocities):
        """Return ring-point states at several epochs.

        Parameters
        ----------
        epochs : sequence of Epoch
            Evaluation epochs (TDB).
        sun_positions : array_like
            BCRS Sun positions in m, shape ``(len(epochs), 3)``; used only at
            the reference epoch of an unprepared ring.
        sun_velocities : array_like
            BCRS Sun velocities in m s⁻¹, shape ``(len(epochs), 3)``.

        Returns
        -------
        positions : numpy.ndarray
            BCRS positions in m, shape ``(len(epochs), point_count, 3)``.
        velocities : numpy.ndarray
            BCRS velocities in m s⁻¹, shape ``(len(epochs), point_count, 3)``.

        Raises
        ------
        ValueError
            If an epoch lies outside the prepropagated interval.
        """
        seconds = np.asarray([((epoch.jd1 - self.reference_jd_tdb) + epoch.jd2) * 86400.0 for epoch in epochs])
        states = np.empty((len(epochs), 2 * self.point_count * 3))
        for mask, solution in ((seconds > 0, self._solution), (seconds < 0, self._backward_solution)):
            if not np.any(mask):
                continue
            selected = seconds[mask]
            if (
                solution is None
                or selected.min() < min(solution.t_min, solution.t_max)
                or selected.max() > max(solution.t_min, solution.t_max)
            ):
                raise ValueError("Ring epoch is outside its prepropagated interval")
            states[mask] = solution(selected).T
        for index in np.flatnonzero(seconds == 0):
            initial = (
                self.circular_states(epochs[index], sun_positions[index], sun_velocities[index])
                if self._initial_states is None
                else self._initial_states
            )
            states[index] = np.concatenate([item.ravel() for item in initial])
        split = self.point_count * 3
        return states[:, :split].reshape(-1, self.point_count, 3), states[:, split:].reshape(-1, self.point_count, 3)

    def states(self, epoch: Epoch, sun_position, sun_velocity):
        """Return ring-point states at one epoch.

        Parameters
        ----------
        epoch : Epoch
            Evaluation epoch (TDB).
        sun_position : array_like
            BCRS Sun position in m, shape ``(3,)``.
        sun_velocity : array_like
            BCRS Sun velocity in m s⁻¹, shape ``(3,)``.

        Returns
        -------
        positions : numpy.ndarray
            BCRS positions in m, shape ``(point_count, 3)``.
        velocities : numpy.ndarray
            BCRS velocities in m s⁻¹, shape ``(point_count, 3)``.

        Raises
        ------
        ValueError
            If ``epoch`` lies outside the prepropagated interval.
        """
        positions, velocities = self.states_many((epoch,), (sun_position,), (sun_velocity,))
        return positions[0], velocities[0]
