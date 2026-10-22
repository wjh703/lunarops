"""Newtonian point-mass, EIH, solar-source, and Earth-tide kernels.

Spherical-harmonic gravity is implemented in the Cython kernel exposed by
:mod:`.gravity`; this module contains only scalar/vector force kernels.

EIH follows Park et al. (2021), Eq. 27, with R_ij = r_j - r_i.
All bodies, including prescribed sources, enter the point-mass Newtonian
auxiliary sums.  The acceleration appearing inside the 1PN EIH term is this
lower-order point-mass acceleration; figure/tide terms are not substituted
without a separate extended-body 1PN derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType

import erfa
import numpy as np

from lunarops._dynamics_core import (
    earth_tide_relative_acceleration,
    eih_correction,
    point_mass_with_cache,
)
from lunarops.base.constants import SECONDS_PER_DAY, C
from lunarops.classes.ephemerides.body_ids import body_name

from ._vector import cross3


@dataclass(frozen=True, slots=True)
class SolarSourceParameters:
    """Validated immutable constants for solar relativistic and SRP terms."""

    gravitational_parameter_m3_s2: float
    radius_m: float = 696_000_000.0
    spin_rate_rad_s: float = np.deg2rad(14.1844) / SECONDS_PER_DAY
    spin_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gamma: float = 1.0
    moment_factor: float = 0.06884
    _angular_momentum: np.ndarray = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self):
        axis = np.asarray(self.spin_axis, dtype=float)
        object.__setattr__(self, "spin_axis", tuple(float(x) for x in axis))
        angular_momentum = self.moment_factor * self.gravitational_parameter_m3_s2 * self.radius_m**2 * self.spin_rate_rad_s * axis
        angular_momentum.setflags(write=False)
        object.__setattr__(self, "_angular_momentum", angular_momentum)

    def lense_thirring_acceleration(self, r, v):
        d = np.linalg.norm(r)
        omega = (
            (1.0 + self.gamma)
            / (2.0 * C**2 * d**3)
            * (-self._angular_momentum + 3.0 * np.dot(self._angular_momentum, r) * r / d**2)
        )
        return 2.0 * cross3(omega, v)

    def radiation_pressure_acceleration(self, r, epsilon):
        d = np.linalg.norm(r)
        return epsilon * self.gravitational_parameter_m3_s2 * r / d**3


@dataclass(frozen=True, slots=True)
class EarthTideParameters:
    """DE440 Earth-tide Love numbers and delays, with delays in days."""

    k20: float = 0.335
    k21: float = 0.277
    k22: float = 0.283
    tau_orb_days: tuple[float, float, float] = (0.077, -0.010, -0.20982243826832672)
    tau_rot_days: tuple[float, float, float] = (0.0, 0.011929122823877609, -0.0007423627529883070)
    earth_rotation_rate_rad_s: float = 7.29211514670698e-5

    def __post_init__(self):
        tau_orb_days = tuple(float(value) for value in self.tau_orb_days)
        tau_rot_days = tuple(float(value) for value in self.tau_rot_days)
        object.__setattr__(self, "tau_orb_days", tau_orb_days)
        object.__setattr__(self, "tau_rot_days", tau_rot_days)


class EarthTideModel:
    """JPL Eq. (35)-(36) Earth tide as a relative Earth-Moon acceleration.

    The delayed-state history supplies batched barycentric states. The
    returned acceleration is Moon minus Earth; EarthTideForce converts it to
    the Moon's barycentric share for the prescribed-Earth propagator.
    """

    def __init__(
        self,
        earth_gravitational_parameter_m3_s2,
        moon_gravitational_parameter_m3_s2,
        *,
        tide_raiser_gm,
        earth_radius_m=6_378_136.6,
        tide_parameters=None,
        tide_raisers=("MOON", "SUN"),
        solar_gm_variation=None,
    ):
        self.earth_gravitational_parameter_m3_s2 = float(earth_gravitational_parameter_m3_s2)
        self.moon_gravitational_parameter_m3_s2 = float(moon_gravitational_parameter_m3_s2)
        self.earth_radius_m = float(earth_radius_m)
        self.tide_parameters = EarthTideParameters() if tide_parameters is None else tide_parameters
        self.tide_raisers = tuple(body_name(name) for name in tide_raisers)
        raiser_gm = {body_name(name): float(value) for name, value in tide_raiser_gm.items()}
        missing = set(self.tide_raisers) - set(raiser_gm)
        if missing:
            raise ValueError(f"Missing Earth-tide raiser GMs: {sorted(missing)}")
        self.tide_raiser_gm = MappingProxyType(raiser_gm)
        self.solar_gm_variation = solar_gm_variation
        # Constants of every evaluation, computed once with the same expressions as before.
        tide = self.tide_parameters
        self._requested = tuple(dict.fromkeys(("EARTH", *self.tide_raisers)))
        self._raiser_rows = tuple(self._requested.index(name) for name in self.tide_raisers)
        self._delay_seconds = tuple(-tide.tau_orb_days[order] * SECONDS_PER_DAY for order in range(3))
        self._rotation_lags = tuple(
            erfa.rz(-tide.earth_rotation_rate_rad_s * tide.tau_rot_days[order] * SECONDS_PER_DAY, np.eye(3))
            for order in range(3)
        )
        self._base_raiser_gm = np.ascontiguousarray([raiser_gm[name] for name in self.tide_raisers], dtype=float)
        self._sun_row = self.tide_raisers.index("SUN") if "SUN" in self.tide_raisers else None
        self._external_rows = {name: index for index, name in enumerate(self.external_names)}

    @property
    def external_names(self) -> tuple[str, ...]:
        """Prescribed bodies of one delayed history query, in the order the history provider reads them."""
        return tuple(name for name in self._requested if name != "MOON")

    @property
    def delay_seconds(self) -> tuple[float, ...]:
        return self._delay_seconds

    def _delayed_states(self, external, relative_state, epoch):
        """Rows of ``self._requested``: prescribed bodies from ``external``, the Moon from integrated history."""
        states = np.empty((len(self._requested), 6), dtype=float)
        for index, name in enumerate(self._requested):
            if name == "MOON":
                states[index] = external[0] + np.asarray(relative_state(epoch), dtype=float)[:6]
            else:
                states[index] = external[self._external_rows[name]]
        return states

    def delayed_vectors(self, epoch, inertial2fixed_matrix, history, delayed_external_states=None):
        """Raiser vectors at the three delayed epochs, in the current Earth-fixed basis, plus raiser GMs.

        Depends only on accepted history and the epoch, not on the candidate Moon state, so one
        ABM step evaluates it once for all of its corrector iterations. With an integrated
        history, the prescribed bodies come from ``delayed_external_states`` (fetched with the
        epoch data); other providers, such as the startup's ephemeris Moon, are queried directly.
        """
        relative_state = getattr(history, "relative_state", None)
        prefetched = delayed_external_states is not None and relative_state is not None
        delayed_vectors = np.empty((len(self.tide_raisers), 3, 3), dtype=float)
        for order in range(3):
            delayed_epoch = epoch.shifted(self._delay_seconds[order])
            states = (
                self._delayed_states(delayed_external_states[order], relative_state, delayed_epoch)
                if prefetched
                else history(self._requested, delayed_epoch)
            )
            delayed_earth = states[0, :3]
            rotation_lag = self._rotation_lags[order]
            for raiser_index, row in enumerate(self._raiser_rows):
                # Eq. (36) combines these vectors with the current Earth-Moon
                # vector, so every operand must use the current fixed-frame basis.
                delayed_vectors[raiser_index, order] = (
                    rotation_lag @ inertial2fixed_matrix @ (states[row, :3] - delayed_earth)
                )
        raiser_gm = self._base_raiser_gm
        if self.solar_gm_variation is not None and self._sun_row is not None:
            raiser_gm = raiser_gm.copy()
            raiser_gm[self._sun_row] *= self.solar_gm_variation.factor(epoch)
        return delayed_vectors, raiser_gm

    def acceleration_from_vectors(self, moon_minus_earth, inertial2fixed_matrix, delayed_vectors, raiser_gm):
        """Relative Earth-Moon tide acceleration for a candidate state and precomputed delayed vectors."""
        p = self.tide_parameters
        fixed_acceleration = earth_tide_relative_acceleration(
            inertial2fixed_matrix @ moon_minus_earth,
            delayed_vectors,
            raiser_gm,
            self.earth_gravitational_parameter_m3_s2,
            self.moon_gravitational_parameter_m3_s2,
            self.earth_radius_m,
            p.k20,
            p.k21,
            p.k22,
        )
        return inertial2fixed_matrix.T @ fixed_acceleration

    def relative_acceleration(self, epoch, earth, moon, current_inertial2fixed_matrix, history):
        """Evaluate delayed Earth tides in the current Earth-fixed frame."""
        matrix = np.ascontiguousarray(current_inertial2fixed_matrix)
        vectors, raiser_gm = self.delayed_vectors(epoch, matrix, history)
        return self.acceleration_from_vectors(moon - earth, matrix, vectors, raiser_gm)

@dataclass(frozen=True, slots=True)
class PointMassGravityEvaluation:
    """Complete Newtonian auxiliary state required by the EIH equations."""

    accelerations_mps2: np.ndarray
    potentials_m2_s2: np.ndarray


@dataclass(frozen=True, slots=True)
class PointMassGravityCache:
    """Newtonian accelerations and potentials among cached source bodies at one epoch."""

    integrated_body_count: int
    prescribed_body_accelerations_mps2: np.ndarray
    prescribed_body_potentials_m2_s2: np.ndarray


def evaluate_newtonian_point_mass_system(body_positions_m, gravitational_parameters_m3_s2, point_mass_gravity_cache):
    """Evaluate all Newtonian accelerations and potentials, reusing fixed source terms."""
    acceleration, potential = point_mass_with_cache(
        body_positions_m,
        gravitational_parameters_m3_s2,
        point_mass_gravity_cache.integrated_body_count,
        point_mass_gravity_cache.prescribed_body_accelerations_mps2,
        point_mass_gravity_cache.prescribed_body_potentials_m2_s2,
    )
    return PointMassGravityEvaluation(acceleration, potential)


def evaluate_eih_correction(
    body_positions_m,
    body_velocities_mps,
    gravitational_parameters_m3_s2,
    newtonian_evaluation,
    target_body_indices,
):
    """Evaluate Eq. 27 EIH corrections only for the requested target rows."""
    return eih_correction(
        body_positions_m,
        body_velocities_mps,
        gravitational_parameters_m3_s2,
        newtonian_evaluation.potentials_m2_s2,
        newtonian_evaluation.accelerations_mps2,
        target_body_indices,
        C**2,
    )
