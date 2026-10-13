"""Prescribed Kuiper belt ring and epoch-dependent solar mass parameter."""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from lunarops.classes.ephemerides import body_state_matrix
from lunarops.classes.time import Epoch

from .bodies import GravitatingBody

AU_M = 149_597_870_700.0
JULIAN_YEAR_DAYS = 365.25


@dataclass(frozen=True, slots=True)
class SolarGmVariation:
    """Fractional GM drift per Julian year, relative to an explicit TDB epoch."""

    fractional_rate_per_year: float
    reference_jd_tdb: float

    def __post_init__(self):
        if not np.all(np.isfinite((self.fractional_rate_per_year, self.reference_jd_tdb))):
            raise ValueError("Solar GM drift and reference epoch must be finite")

    def factor(self, epoch: Epoch) -> float:
        years = ((epoch.jd1 - self.reference_jd_tdb) + epoch.jd2) / JULIAN_YEAR_DAYS
        factor = 1.0 + self.fractional_rate_per_year * years
        if factor <= 0:
            raise ValueError("Solar GM must remain positive over the propagation interval")
        return factor


@dataclass(frozen=True, slots=True)
class KuiperBeltRing:
    """Equal mass points initially circular in the fixed J2000 ecliptic.

    The radius, mass and count follow DE440. Each point is prepropagated in
    BCRS under prescribed sources, without point-to-point attraction, as in
    Pitjeva & Pitjev (2018). Full JPL ring-point initial conditions are not
    distributed in the BSP; initial phase remains explicit.
    """

    point_gm_m3_s2: float
    solar_gm_m3_s2: float
    radius_m: float
    point_count: int
    obliquity_rad: float
    reference_jd_tdb: float
    phase_rad: float = 0.0
    _solution: object = field(default=None, init=False, repr=False, compare=False)
    _backward_solution: object = field(default=None, init=False, repr=False, compare=False)
    _initial_states: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        values = (self.point_gm_m3_s2, self.solar_gm_m3_s2, self.radius_m,
                  self.obliquity_rad, self.reference_jd_tdb, self.phase_rad)
        if not np.all(np.isfinite(values)) or min(values[:3]) <= 0 or self.point_count < 3:
            raise ValueError("Ring masses/radius must be positive and ring parameters finite")

    @property
    def bodies(self):
        return tuple(GravitatingBody(f"KBO_RING_{i + 1:02d}", self.point_gm_m3_s2)
                     for i in range(self.point_count))

    def circular_states(self, epoch: Epoch, sun_position, sun_velocity):
        omega = np.sqrt(self.solar_gm_m3_s2 / self.radius_m**3)
        elapsed = ((epoch.jd1 - self.reference_jd_tdb) + epoch.jd2) * 86400.0
        angles = self.phase_rad + np.arange(self.point_count) * (2 * np.pi / self.point_count) + omega * elapsed
        c, s = np.cos(angles), np.sin(angles)
        ce, se = np.cos(self.obliquity_rad), np.sin(self.obliquity_rad)
        position = self.radius_m * np.column_stack((c, ce * s, se * s)) + sun_position
        velocity = self.radius_m * omega * np.column_stack((-s, ce * c, se * c)) + sun_velocity
        return position, velocity

    def prepare(self, ephemeris, source_names, source_gm, initial_epoch, duration_s, solar_gm_variation):
        """Precompute independent ring trajectories; Moon remains the propagated target."""
        reference = Epoch(self.reference_jd_tdb, 0.0, "tdb")
        sun = source_names.index("SUN")
        sun_state = body_state_matrix(ephemeris, ("SUN",), reference)[0]
        position, velocity = self.circular_states(reference, sun_state[:3], sun_state[3:])
        if solar_gm_variation is not None:
            velocity = sun_state[3:] + (velocity - sun_state[3:]) * np.sqrt(solar_gm_variation.factor(reference))
        object.__setattr__(self, "_initial_states", (position, velocity))
        initial = np.concatenate((position.ravel(), velocity.ravel()))
        source_gm = np.asarray(source_gm, dtype=float)

        def rhs(seconds, state):
            epoch = reference.shifted(float(seconds))
            sources = body_state_matrix(ephemeris, source_names, epoch)[:, :3]
            gm = source_gm.copy()
            if solar_gm_variation is not None:
                gm[sun] *= solar_gm_variation.factor(epoch)
            points = state[:self.point_count * 3].reshape(-1, 3)
            delta = sources[None, :, :] - points[:, None, :]
            distances = np.linalg.norm(delta, axis=2)
            acceleration = np.sum(delta * (gm[None, :] / distances**3)[:, :, None], axis=1)
            return np.concatenate((state[self.point_count * 3:], acceleration.ravel()))

        offsets = (reference.seconds_until(initial_epoch), reference.seconds_until(initial_epoch) + duration_s)
        # Cover the requested interval on both sides of the independent ring
        # initial epoch. No integration step is taken beyond these endpoints.
        for final, attribute in ((max(0, *offsets), "_solution"), (min(0, *offsets), "_backward_solution")):
            if final == 0:
                continue
            solution = solve_ivp(rhs, (0, final), initial, method="DOP853", dense_output=True,
                                 rtol=3e-14,
                                 atol=np.r_[np.full(self.point_count * 3, .01), np.full(self.point_count * 3, 1e-8)],
                                 max_step=14 * 86400.0)
            if not solution.success:
                raise RuntimeError(f"KBO ring prepropagation failed: {solution.message}")
            object.__setattr__(self, attribute, solution.sol)

    def states(self, epoch: Epoch, sun_position, sun_velocity):
        seconds = ((epoch.jd1 - self.reference_jd_tdb) + epoch.jd2) * 86400.0
        if seconds == 0:
            return self.circular_states(epoch, sun_position, sun_velocity) if self._initial_states is None else self._initial_states
        solution = self._solution if seconds > 0 else self._backward_solution
        if solution is None or not min(solution.t_min, solution.t_max) <= seconds <= max(solution.t_min, solution.t_max):
            raise ValueError("Ring epoch is outside its prepropagated interval")
        state = solution(seconds)
        split = self.point_count * 3
        return state[:split].reshape(-1, 3), state[split:].reshape(-1, 3)
