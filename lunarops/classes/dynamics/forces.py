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

from lunarops._dynamics_core import (
    earth_tide_relative_acceleration,
    eih_correction,
    point_mass_symmetric,
    point_mass_with_cache,
)
from lunarops.base.array_validation import finite_array, rotation_matrix
from lunarops.base.constants import C
from lunarops.classes.ephemerides.body_ids import body_name

SECONDS_PER_DAY = 86400.0
_IDENTITY3 = np.eye(3)
_IDENTITY3.setflags(write=False)


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

    def lense_thirring_acceleration(self, relative_position, relative_velocity):
        r = np.asarray(relative_position, dtype=float)
        v = np.asarray(relative_velocity, dtype=float)
        d = np.linalg.norm(r)
        if r.shape != (3,) or v.shape != (3,) or not np.all(np.isfinite((r, v))) or d == 0:
            raise ValueError("Solar LT state must be finite and nonzero.")
        omega = (
            (1.0 + self.gamma)
            / (2.0 * C**2 * d**3)
            * (-self._angular_momentum + 3.0 * np.dot(self._angular_momentum, r) * r / d**2)
        )
        return 2.0 * np.cross(omega, v)

    def radiation_pressure_acceleration(self, relative_position, epsilon):
        r = np.asarray(relative_position, dtype=float)
        d = np.linalg.norm(r)
        if r.shape != (3,) or not np.all(np.isfinite(r)) or d == 0 or not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("Solar radiation pressure state and epsilon must be valid.")
        return float(epsilon) * self.gravitational_parameter_m3_s2 * r / d**3


def _passive_rotation_z(angle_rad):
    """Return the passive rotation about z used by the DE430 equations."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))


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

    ``history_provider`` supplies batched barycentric states at delayed epochs. The
    returned acceleration is the complete Moon/Earth relative acceleration
    used by the prescribed-Earth lunar propagator.
    """

    def __init__(
        self,
        earth_gravitational_parameter_m3_s2,
        moon_gravitational_parameter_m3_s2,
        state_provider,
        earth_inertial2fixed_matrix_provider=None,
        *,
        earth_radius_m=6_378_136.6,
        tide_parameters=None,
        tide_raisers=("MOON", "SUN"),
        tide_raiser_gm=None,
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
        if not callable(state_provider):
            raise TypeError("Earth-tide state_provider must be callable.")
        self.state_provider = state_provider
        self.earth_inertial2fixed_matrix_provider = earth_inertial2fixed_matrix_provider
        self.tide_parameters = EarthTideParameters() if tide_parameters is None else tide_parameters
        if not isinstance(self.tide_parameters, EarthTideParameters):
            raise TypeError("tide_parameters must be an EarthTideParameters object.")
        self.tide_raisers = tuple(body_name(name) for name in tide_raisers)
        if not self.tide_raisers or any(not name for name in self.tide_raisers):
            raise ValueError("Earth tide must have at least one nonempty tide raiser.")
        if len(self.tide_raisers) != len(set(self.tide_raisers)):
            raise ValueError("Earth-tide raisers must be unique.")
        if tide_raiser_gm is None:
            raise ValueError("tide_raiser_gm must provide a positive GM for every tide raiser.")
        raiser_gm = {body_name(name): float(value) for name, value in tide_raiser_gm.items()}
        if any(not np.isfinite(value) or value <= 0 for value in raiser_gm.values()):
            raise ValueError("Earth-tide raiser GMs must be positive and finite.")
        missing = set(self.tide_raisers) - set(raiser_gm)
        if missing:
            raise ValueError(f"Missing Earth-tide raiser GMs: {sorted(missing)}")
        self.tide_raiser_gm = MappingProxyType(raiser_gm)

    def _load_delayed_states(self, names, epoch, history=None):
        provider = self.state_provider if history is None else history
        states = np.asarray(provider(names, epoch), dtype=float)
        if states.shape != (len(names), 6) or not np.all(np.isfinite(states)):
            raise ValueError("Earth-tide history must return a finite (N,6) state matrix")
        return states

    def _inertial2fixed_matrix(self, epoch):
        if self.earth_inertial2fixed_matrix_provider is None:
            return _IDENTITY3
        return rotation_matrix(
            self.earth_inertial2fixed_matrix_provider(epoch),
            name="earth_tide_inertial2fixed_matrix",
        )

    def relative_acceleration(
        self,
        epoch,
        earth_position,
        moon_position,
        current_inertial2fixed_matrix=None,
        *,
        history=None,
    ):
        """Evaluate delayed Earth tides in the current Earth-fixed frame."""
        earth = np.asarray(earth_position, dtype=float)
        moon = np.asarray(moon_position, dtype=float)
        if earth.shape != (3,) or moon.shape != (3,) or not np.all(np.isfinite((earth, moon))):
            raise ValueError("Earth-tide positions must be finite three-vectors.")
        inertial2fixed_matrix = (
            self._inertial2fixed_matrix(epoch)
            if current_inertial2fixed_matrix is None
            else rotation_matrix(current_inertial2fixed_matrix, name="current_inertial2fixed_matrix")
        )
        r = inertial2fixed_matrix @ (moon - earth)
        radius = np.linalg.norm(r)
        if radius == 0:
            raise ValueError("Earth-tide Earth-Moon position must be nonzero.")
        p = self.tide_parameters
        requested = tuple(dict.fromkeys(("EARTH", *self.tide_raisers)))
        delayed_states = []
        for order in range(3):
            delayed_epoch = epoch.shifted(-p.tau_orb_days[order] * SECONDS_PER_DAY)
            state_matrix = self._load_delayed_states(requested, delayed_epoch, history)
            state_by_name = {name: state_matrix[index, :3] for index, name in enumerate(requested)}
            delayed_earth = state_by_name["EARTH"]
            rotation_lag = _passive_rotation_z(
                -p.earth_rotation_rate_rad_s * p.tau_rot_days[order] * SECONDS_PER_DAY
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
        fixed_acceleration = earth_tide_relative_acceleration(
            np.ascontiguousarray(r),
            np.ascontiguousarray(delayed_vectors),
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

    def __post_init__(self):
        acceleration = np.array(self.accelerations_mps2, dtype=float, copy=True)
        if acceleration.ndim != 2 or acceleration.shape[1] != 3 or not np.all(np.isfinite(acceleration)):
            raise ValueError("accelerations_mps2 must be a finite (N,3) array")
        acceleration = np.ascontiguousarray(acceleration)
        acceleration.setflags(write=False)
        potential = finite_array(
            self.potentials_m2_s2,
            size=len(acceleration),
            name="potentials_m2_s2",
            readonly=True,
        )
        object.__setattr__(self, "accelerations_mps2", acceleration)
        object.__setattr__(self, "potentials_m2_s2", potential)


@dataclass(frozen=True, slots=True)
class PointMassGravityCache:
    """Newtonian accelerations and potentials among cached source bodies at one epoch."""

    integrated_body_count: int
    prescribed_body_accelerations_mps2: np.ndarray
    prescribed_body_potentials_m2_s2: np.ndarray

    def __post_init__(self):
        count = int(self.integrated_body_count)
        if count < 0:
            raise ValueError("integrated_body_count must be nonnegative")
        acceleration = np.array(self.prescribed_body_accelerations_mps2, dtype=float, copy=True)
        if acceleration.ndim != 2 or acceleration.shape[1] != 3 or not np.all(np.isfinite(acceleration)):
            raise ValueError("prescribed_body_accelerations_mps2 must be a finite (N,3) array")
        acceleration = np.ascontiguousarray(acceleration)
        acceleration.setflags(write=False)
        potential = finite_array(
            self.prescribed_body_potentials_m2_s2,
            size=len(acceleration),
            name="prescribed_body_potentials_m2_s2",
            readonly=True,
        )
        object.__setattr__(self, "integrated_body_count", count)
        object.__setattr__(self, "prescribed_body_accelerations_mps2", acceleration)
        object.__setattr__(self, "prescribed_body_potentials_m2_s2", potential)


def _newtonian_acceleration_and_potential(body_positions_m, gravitational_parameters_m3_s2):
    r = np.ascontiguousarray(body_positions_m, dtype=float)
    mu = np.ascontiguousarray(gravitational_parameters_m3_s2, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3 or mu.shape != (len(r),):
        raise ValueError("Expected positions (N,3) and GM (N,).")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("Positions must be finite and GM positive.")
    return point_mass_symmetric(r, mu)


def build_point_mass_gravity_cache(body_positions_m, gravitational_parameters_m3_s2, *, integrated_body_count):
    """Cache source/source gravity, excluding the first integrated bodies."""
    r = np.asarray(body_positions_m, dtype=float)
    mu = np.asarray(gravitational_parameters_m3_s2, dtype=float)
    count = int(integrated_body_count)
    if count < 0 or count > len(r):
        raise ValueError("integrated_body_count is outside the body array")
    external_acceleration, external_potential = _newtonian_acceleration_and_potential(r[count:], mu[count:])
    return PointMassGravityCache(count, external_acceleration, external_potential)


def evaluate_newtonian_point_mass_system(body_positions_m, gravitational_parameters_m3_s2, point_mass_gravity_cache=None):
    """Evaluate all Newtonian accelerations and potentials, reusing fixed source terms."""
    r = np.asarray(body_positions_m, dtype=float)
    mu = np.asarray(gravitational_parameters_m3_s2, dtype=float)
    if point_mass_gravity_cache is None:
        acceleration, potential = _newtonian_acceleration_and_potential(r, mu)
        return PointMassGravityEvaluation(acceleration, potential)
    if not isinstance(point_mass_gravity_cache, PointMassGravityCache):
        raise TypeError("point_mass_gravity_cache must be a PointMassGravityCache")
    integrated_body_count = point_mass_gravity_cache.integrated_body_count
    if integrated_body_count > len(r):
        raise ValueError("Point-mass gravity cache has too many integrated bodies")
    if r.ndim != 2 or r.shape[1] != 3 or mu.shape != (len(r),):
        raise ValueError("Expected positions (N,3) and GM (N,).")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("Positions must be finite and GM positive.")
    if point_mass_gravity_cache.prescribed_body_accelerations_mps2.shape != (
        len(r) - integrated_body_count,
        3,
    ):
        raise ValueError("Point-mass gravity cache does not match the body array")

    acceleration, potential = point_mass_with_cache(
        np.ascontiguousarray(r),
        np.ascontiguousarray(mu),
        integrated_body_count,
        np.ascontiguousarray(point_mass_gravity_cache.prescribed_body_accelerations_mps2),
        np.ascontiguousarray(point_mass_gravity_cache.prescribed_body_potentials_m2_s2),
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
    r, v, mu = (
        np.array(value, dtype=float, copy=True)
        for value in (body_positions_m, body_velocities_mps, gravitational_parameters_m3_s2)
    )
    targets = np.asarray(target_body_indices)
    if r.ndim != 2 or r.shape[1] != 3 or v.shape != r.shape or mu.shape != (len(r),):
        raise ValueError("Expected positions/velocities (N,3) and GM (N,).")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("Positions, velocities, and GM must be finite; GM must be positive.")
    if targets.ndim != 1 or (targets.size and not np.issubdtype(targets.dtype, np.integer)):
        raise ValueError("target_body_indices must be a one-dimensional integer array")
    targets = np.asarray(targets, dtype=np.intp)
    if not isinstance(newtonian_evaluation, PointMassGravityEvaluation):
        raise TypeError("newtonian_evaluation must be a PointMassGravityEvaluation")
    if len(targets) == 0:
        return np.zeros_like(r)

    return eih_correction(
        np.ascontiguousarray(r),
        np.ascontiguousarray(v),
        np.ascontiguousarray(mu),
        np.ascontiguousarray(newtonian_evaluation.potentials_m2_s2),
        np.ascontiguousarray(newtonian_evaluation.accelerations_mps2),
        np.ascontiguousarray(targets, dtype=np.intp),
        C**2,
    )
