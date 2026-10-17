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

import numpy as np
import erfa

from lunarops._dynamics_core import (
    earth_tide_relative_acceleration,
    eih_correction,
    point_mass_symmetric_batch,
    point_mass_with_cache,
)
from lunarops.base.constants import C, SECONDS_PER_DAY
from lunarops.classes.ephemerides.body_ids import body_name


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
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("Solar spin axis must be a finite three-vector.")
        if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-14, rtol=0):
            raise ValueError("Solar spin axis must have unit length.")
        if not np.all(
            np.isfinite(
                (
                    self.gravitational_parameter_m3_s2,
                    self.radius_m,
                    self.spin_rate_rad_s,
                    self.gamma,
                    self.moment_factor,
                )
            )
        ):
            raise ValueError("Solar source parameters must be finite.")
        if self.gravitational_parameter_m3_s2 <= 0 or self.radius_m <= 0:
            raise ValueError("Solar gravitational parameter and radius must be positive.")
        if self.spin_rate_rad_s < 0 or self.moment_factor <= 0:
            raise ValueError("Solar spin rate must be nonnegative and moment factor positive.")
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
        return 2.0 * np.cross(omega, v)

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
        if len(tau_orb_days) != 3 or len(tau_rot_days) != 3:
            raise ValueError("Earth-tide delays must contain orders 0, 1, and 2.")
        values = (self.k20, self.k21, self.k22, self.earth_rotation_rate_rad_s, *tau_orb_days, *tau_rot_days)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Earth-tide parameters must be finite.")
        if any(value < 0 for value in (self.k20, self.k21, self.k22)):
            raise ValueError("Earth-tide Love numbers must be nonnegative.")
        if self.earth_rotation_rate_rad_s <= 0:
            raise ValueError("Earth rotation rate must be positive.")
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
        if not np.isfinite(self.earth_gravitational_parameter_m3_s2) or self.earth_gravitational_parameter_m3_s2 <= 0:
            raise ValueError("Earth gravitational parameter must be positive and finite.")
        if not np.isfinite(self.moon_gravitational_parameter_m3_s2) or self.moon_gravitational_parameter_m3_s2 <= 0:
            raise ValueError("Moon gravitational parameter must be positive and finite.")
        if not np.isfinite(self.earth_radius_m) or self.earth_radius_m <= 0:
            raise ValueError("Earth radius must be positive and finite.")
        self.tide_parameters = EarthTideParameters() if tide_parameters is None else tide_parameters
        self.tide_raisers = tuple(body_name(name) for name in tide_raisers)
        if not self.tide_raisers or any(not name for name in self.tide_raisers):
            raise ValueError("Earth tide must have at least one nonempty tide raiser.")
        if len(self.tide_raisers) != len(set(self.tide_raisers)):
            raise ValueError("Earth-tide raisers must be unique.")
        raiser_gm = {body_name(name): float(value) for name, value in tide_raiser_gm.items()}
        if any(not np.isfinite(value) or value <= 0 for value in raiser_gm.values()):
            raise ValueError("Earth-tide raiser GMs must be positive and finite.")
        missing = set(self.tide_raisers) - set(raiser_gm)
        if missing:
            raise ValueError(f"Missing Earth-tide raiser GMs: {sorted(missing)}")
        self.tide_raiser_gm = MappingProxyType(raiser_gm)
        self.solar_gm_variation = solar_gm_variation

    def relative_acceleration(self, epoch, earth, moon, current_inertial2fixed_matrix, history):
        """Evaluate delayed Earth tides in the current Earth-fixed frame."""
        inertial2fixed_matrix = np.ascontiguousarray(current_inertial2fixed_matrix)
        r = inertial2fixed_matrix @ (moon - earth)
        p = self.tide_parameters
        requested = tuple(dict.fromkeys(("EARTH", *self.tide_raisers)))
        delayed_states = []
        for order in range(3):
            delayed_epoch = epoch.shifted(-p.tau_orb_days[order] * SECONDS_PER_DAY)
            state_matrix = history(requested, delayed_epoch)
            state_by_name = {name: state_matrix[index, :3] for index, name in enumerate(requested)}
            delayed_earth = state_by_name["EARTH"]
            rotation_lag = erfa.rz(
                -p.earth_rotation_rate_rad_s * p.tau_rot_days[order] * SECONDS_PER_DAY, np.eye(3)
            )
            delayed_states.append((delayed_earth, rotation_lag, state_by_name))

        delayed_vectors = np.empty((len(self.tide_raisers), 3, 3), dtype=float)
        for raiser_index, raiser in enumerate(self.tide_raisers):
            for order, (delayed_earth, rotation_lag, state_by_name) in enumerate(delayed_states):
                source = state_by_name[raiser]
                # Eq. (36) combines these vectors with the current Earth-Moon
                # vector, so every operand must use the current fixed-frame basis.
                delayed_vectors[raiser_index, order] = (
                    rotation_lag @ inertial2fixed_matrix @ (source - delayed_earth)
                )
        raiser_gm = np.ascontiguousarray([self.tide_raiser_gm[name] for name in self.tide_raisers], dtype=float)
        if self.solar_gm_variation is not None and "SUN" in self.tide_raisers:
            raiser_gm[self.tide_raisers.index("SUN")] *= self.solar_gm_variation.factor(epoch)
        fixed_acceleration = earth_tide_relative_acceleration(
            r,
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


def build_point_mass_gravity_cache(body_positions_m, gravitational_parameters_m3_s2, *, integrated_body_count):
    return build_point_mass_gravity_cache_batch(
        np.asarray(body_positions_m)[None], np.asarray(gravitational_parameters_m3_s2)[None],
        integrated_body_count=integrated_body_count,
    )[0]


def build_point_mass_gravity_cache_batch(
    body_positions_m, gravitational_parameters_m3_s2, *, integrated_body_count
):
    """Build source/source caches for a batch of epochs in one native call."""
    count = integrated_body_count
    accelerations, potentials = point_mass_symmetric_batch(
        np.ascontiguousarray(body_positions_m[:, count:]),
        np.ascontiguousarray(gravitational_parameters_m3_s2[:, count:]),
    )
    return tuple(
        PointMassGravityCache(count, acceleration, potential)
        for acceleration, potential in zip(accelerations, potentials, strict=True)
    )


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
