"""SI Newtonian, first post-Newtonian EIH, and axisymmetric J2 kernels.

Spherical-harmonic gravity is implemented in :mod:`.gravity` on top of
``pyshtools.SHGravCoeffs``; this module contains only scalar/vector kernels.

EIH follows Park et al. (2021), Eq. 27, with R_ij = r_j - r_i.
All bodies, including prescribed sources, enter the point-mass Newtonian
auxiliary sums.  The acceleration appearing inside the 1PN EIH term is this
lower-order point-mass acceleration; figure/tide terms are not substituted
without a separate extended-body 1PN derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from lunarops._dynamics_core import point_mass_symmetric
from lunarops.base.constants import C
from lunarops.classes.ephemerides.body_ids import body_name

SECONDS_PER_DAY = 86400.0
_Z_AXIS = np.array((0.0, 0.0, 1.0))
_IDENTITY3 = np.eye(3)
_IDENTITY3.setflags(write=False)


def solar_lense_thirring_acceleration(
    relative_position,
    relative_velocity,
    solar_gm,
    solar_radius_m,
    solar_spin_rate_rad_s,
    solar_pole,
    *,
    gamma=1.0,
    solar_moment_factor=0.06884,
):
    """DE440 solar Lense-Thirring acceleration (Park et al. Eq. 37-39)."""
    r = np.asarray(relative_position, float)
    v = np.asarray(relative_velocity, float)
    pole = np.asarray(solar_pole, float)
    if r.shape != (3,) or v.shape != (3,) or pole.shape != (3,) or not np.all(np.isfinite((r, v, pole))):
        raise ValueError("Solar LT inputs must be finite three-vectors.")
    distance = np.linalg.norm(r)
    if distance == 0 or not np.isclose(np.linalg.norm(pole), 1.0, atol=1e-14, rtol=0):
        raise ValueError("Solar LT position must be nonzero and pole unit length.")
    values = (solar_gm, solar_radius_m, solar_spin_rate_rad_s, gamma, solar_moment_factor)
    if not np.all(np.isfinite(values)) or solar_gm <= 0 or solar_radius_m <= 0:
        raise ValueError("Solar LT constants must be finite and positive.")
    gj = solar_moment_factor * solar_gm * solar_radius_m**2 * solar_spin_rate_rad_s * pole
    omega = (1.0 + gamma) / (2.0 * C**2 * distance**3) * (-gj + 3.0 * np.dot(gj, r) * r / distance**2)
    return 2.0 * np.cross(omega, v)


def solar_radiation_pressure_acceleration(relative_position, solar_gm, epsilon):
    """DE440 inverse-square SRP, positive along the Sun-to-body vector."""
    r = np.asarray(relative_position, float)
    if r.shape != (3,) or not np.all(np.isfinite(r)):
        raise ValueError("Solar radiation pressure position must be finite.")
    distance = np.linalg.norm(r)
    if distance == 0 or not np.isfinite(solar_gm) or solar_gm <= 0:
        raise ValueError("Solar radiation pressure position and GM must be valid.")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Solar radiation pressure epsilon must be finite and nonnegative.")
    return float(epsilon) * float(solar_gm) * r / distance**3


@dataclass(frozen=True, slots=True)
class SolarForceParameters:
    """Validated immutable constants for solar relativistic and SRP terms."""

    gm_m3ps2: float
    radius_m: float = 696_000_000.0
    spin_rate_rad_s: float = np.deg2rad(14.1844) / SECONDS_PER_DAY
    spin_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gamma: float = 1.0
    moment_factor: float = 0.06884
    _axis: np.ndarray = dataclass_field(init=False, repr=False, compare=False)
    _angular_momentum: np.ndarray = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self):
        axis = np.asarray(self.spin_axis, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("Solar spin axis must be a finite three-vector.")
        if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-14, rtol=0):
            raise ValueError("Solar spin axis must have unit length.")
        if (
            not np.all(
                np.isfinite((self.gm_m3ps2, self.radius_m, self.spin_rate_rad_s, self.gamma, self.moment_factor))
            )
            or self.gm_m3ps2 <= 0
            or self.radius_m <= 0
        ):
            raise ValueError("Solar force constants must be finite and positive.")
        object.__setattr__(self, "spin_axis", tuple(float(x) for x in axis))
        axis = np.array(axis, dtype=float)
        axis.setflags(write=False)
        object.__setattr__(self, "_axis", axis)
        angular_momentum = self.moment_factor * self.gm_m3ps2 * self.radius_m**2 * self.spin_rate_rad_s * axis
        angular_momentum.setflags(write=False)
        object.__setattr__(self, "_angular_momentum", angular_momentum)

    @property
    def axis_array(self):
        return self._axis

    def lense_thirring(self, relative_position, relative_velocity):
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

    def radiation_pressure(self, relative_position, epsilon):
        r = np.asarray(relative_position, dtype=float)
        d = np.linalg.norm(r)
        if r.shape != (3,) or not np.all(np.isfinite(r)) or d == 0 or not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("Solar radiation pressure state and epsilon must be valid.")
        return float(epsilon) * self.gm_m3ps2 * r / d**3


def _rotate_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


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
        values = (self.k20, self.k21, self.k22, self.earth_rotation_rate_rad_s, *self.tau_orb_days, *self.tau_rot_days)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Earth-tide parameters must be finite.")
        if any(value < 0 for value in (self.k20, self.k21, self.k22)):
            raise ValueError("Earth-tide Love numbers must be nonnegative.")
        if self.earth_rotation_rate_rad_s <= 0:
            raise ValueError("Earth rotation rate must be positive.")
        if len(self.tau_orb_days) != 3 or len(self.tau_rot_days) != 3:
            raise ValueError("Earth-tide delays must contain orders 0, 1, and 2.")


class EarthTideModel:
    """JPL Eq. (35)-(36) Earth tide as a relative Earth-Moon acceleration.

    ``initial_history`` supplies batched barycentric states at delayed epochs. The
    returned acceleration is split by :meth:`split_relative_acceleration` so
    the integrated Earth-Moon barycenter is unchanged.
    """

    def __init__(
        self,
        earth_gm,
        moon_gm,
        initial_history,
        earth_frame_provider=None,
        *,
        earth_radius_m=6_378_136.6,
        parameters=None,
        tide_raisers=("MOON", "SUN"),
        tide_raiser_gm=None,
    ):
        self.earth_gm = float(earth_gm)
        self.moon_gm = float(moon_gm)
        self.earth_radius_m = float(earth_radius_m)
        if not np.isfinite(self.earth_gm) or self.earth_gm <= 0:
            raise ValueError("Earth GM must be positive and finite.")
        if not np.isfinite(self.moon_gm) or self.moon_gm <= 0:
            raise ValueError("Moon GM must be positive and finite.")
        if not np.isfinite(self.earth_radius_m) or self.earth_radius_m <= 0:
            raise ValueError("Earth radius must be positive and finite.")
        if not callable(initial_history):
            raise TypeError("Earth-tide initial_history must be callable.")
        self.initial_history = initial_history
        self.earth_frame_provider = earth_frame_provider
        self.parameters = EarthTideParameters() if parameters is None else parameters
        if not isinstance(self.parameters, EarthTideParameters):
            raise TypeError("parameters must be an EarthTideParameters object.")
        self.tide_raisers = tuple(body_name(name) for name in tide_raisers)
        if not self.tide_raisers or any(not name for name in self.tide_raisers):
            raise ValueError("Earth tide must have at least one nonempty tide raiser.")
        self._raiser_gm = {}
        if tide_raiser_gm is not None:
            self.set_tide_raiser_gm(tide_raiser_gm)

    @staticmethod
    def split_relative_acceleration(relative_acceleration, earth_gm, moon_gm):
        """Return Earth/Moon accelerations preserving their barycenter."""
        delta = np.asarray(relative_acceleration, dtype=float)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("Relative tide acceleration must be a finite three-vector.")
        total_gm = float(earth_gm) + float(moon_gm)
        if not np.isfinite(total_gm) or total_gm <= 0:
            raise ValueError("Earth and Moon GM must have a positive finite sum.")
        return (-float(moon_gm) / total_gm * delta, float(earth_gm) / total_gm * delta)

    def _states(self, names, epoch, history=None):
        provider = self.initial_history if history is None else history
        states = np.asarray(provider(names, epoch), dtype=float)
        if states.shape != (len(names), 6) or not np.all(np.isfinite(states)):
            raise ValueError("Earth-tide history must return a finite (N,6) state matrix")
        return states

    def _frame(self, epoch):
        if self.earth_frame_provider is None:
            return _IDENTITY3
        frame = np.asarray(self.earth_frame_provider(epoch), dtype=float)
        if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
            raise ValueError("Earth-tide frame provider must return a finite 3x3 matrix.")
        return frame

    def relative_acceleration(self, epoch, earth_position, moon_position, current_frame=None, *, history=None):
        """Evaluate delayed Earth tides in the current Earth-fixed frame."""
        earth = np.asarray(earth_position, dtype=float)
        moon = np.asarray(moon_position, dtype=float)
        if earth.shape != (3,) or moon.shape != (3,) or not np.all(np.isfinite((earth, moon))):
            raise ValueError("Earth-tide positions must be finite three-vectors.")
        frame = self._frame(epoch) if current_frame is None else np.asarray(current_frame, dtype=float)
        if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
            raise ValueError("Current Earth-tide frame must be a finite 3x3 matrix.")
        r = frame @ (moon - earth)
        radius = np.linalg.norm(r)
        if radius == 0:
            raise ValueError("Earth-tide Earth-Moon position must be nonzero.")
        rho = np.array((r[0], r[1], 0.0))
        z = r[2]
        p = self.parameters
        total = np.zeros(3)
        requested = tuple(dict.fromkeys(("EARTH", *self.tide_raisers)))
        delayed = []
        for order in range(3):
            delayed_epoch = epoch.shifted(-p.tau_orb_days[order] * SECONDS_PER_DAY)
            state_matrix = self._states(requested, delayed_epoch, history)
            state_by_name = {name: state_matrix[index, :3] for index, name in enumerate(requested)}
            delayed_earth = state_by_name["EARTH"]
            rotation_lag = _rotate_z(-p.earth_rotation_rate_rad_s * p.tau_rot_days[order] * SECONDS_PER_DAY)
            delayed.append((delayed_earth, rotation_lag, state_by_name))
        for raiser in self.tide_raisers:
            order_vectors = []
            for delayed_earth, rotation_lag, state_by_name in delayed:
                source = state_by_name[raiser]
                # Eq. (36) combines these vectors with the current Earth-Moon
                # vector, so every operand must use the current fixed-frame basis.
                order_vectors.append(rotation_lag @ frame @ (source - delayed_earth))
            r0, r1, _r2 = order_vectors
            rho0, rho1, rho2 = (np.array((v[0], v[1], 0.0)) for v in order_vectors)
            z0, z1 = r0[2], r1[2]
            norm0, norm1, norm2 = (np.linalg.norm(v) for v in order_vectors)
            if min(norm0, norm1, norm2) == 0:
                raise ValueError("Earth-tide raiser position must be nonzero.")
            dot1 = np.dot(rho, rho1)
            dot2 = np.dot(rho, rho2)
            common = r / radius**2
            t0 = (
                2.0 * z * z0**2 * np.array((0.0, 0.0, 1.0))
                + np.dot(rho0, rho0) * rho
                - 5.0 * ((z * z0) ** 2 + 0.5 * np.dot(rho, rho) * np.dot(rho0, rho0)) * common
                + norm0**2 * r
            ) / norm0**5
            t1 = (
                2.0 * dot1 * z1 * np.array((0.0, 0.0, 1.0)) + 2.0 * z * z1 * rho1 - 10.0 * z * z1 * dot1 * common
            ) / norm1**5
            t2 = (
                2.0 * dot2 * rho2
                - np.dot(rho2, rho2) * rho
                - 5.0 * (dot2**2 - 0.5 * np.dot(rho, rho) * np.dot(rho2, rho2)) * common
            ) / norm2**5
            tide_mu = float(getattr(self, "_raiser_gm", {}).get(raiser, 0.0))
            if tide_mu <= 0:
                raise ValueError(f"Missing positive GM for Earth-tide raiser {raiser!r}.")
            factor = 1.5 * (self.earth_gm + self.moon_gm) / self.earth_gm
            factor *= tide_mu * self.earth_radius_m**5 / radius**5
            total += factor * (p.k20 * t0 + p.k21 * t1 + p.k22 * t2)
        return frame.T @ total

    def set_tide_raiser_gm(self, values):
        values = {body_name(name): float(value) for name, value in values.items()}
        if any(not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("Earth-tide raiser GMs must be positive and finite.")
        self._raiser_gm = values

    def accelerations(self, epoch, earth_position, moon_position, current_frame=None):
        """Return relative tide acceleration and its diagnostic split."""
        delta = self.relative_acceleration(epoch, earth_position, moon_position, current_frame)
        return self.split_relative_acceleration(delta, self.earth_gm, self.moon_gm), delta


def extended_body_pair_accelerations(
    source_index,
    target_index,
    barycentric_positions_m,
    gravitational_parameters_m3_s2,
    field,
    inertial_to_source_body_fixed,
):
    """Return direct target and recoil accelerations for one extended body.

    ``barycentric_positions_m`` are expressed in one common Newtonian
    barycentric inertial frame. Only their difference is used, so the choice
    of inertial origin cancels exactly. ``inertial_to_source_body_fixed`` is a
    rotation about the source body's center, not a translation.
    """
    positions = np.asarray(barycentric_positions_m, dtype=float)
    mus = np.asarray(gravitational_parameters_m3_s2, dtype=float)
    frame = np.asarray(inertial_to_source_body_fixed, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or mus.shape != (len(positions),):
        raise ValueError("Positions must be (N,3) and gravitational parameters must be (N,).")
    if frame.shape != (3, 3) or not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-12):
        raise ValueError("inertial_to_source_body_fixed must be an orthogonal rotation matrix.")
    relative_inertial = positions[target_index] - positions[source_index]
    relative_body_fixed = frame @ relative_inertial
    direct_inertial = frame.T @ field.figure_acceleration(relative_body_fixed)
    recoil = -(mus[target_index] / mus[source_index]) * direct_inertial
    return direct_inertial, recoil


@dataclass(frozen=True, slots=True)
class NewtonianPointMassEvaluation:
    """Complete Newtonian auxiliary state required by the EIH equations."""

    accelerations_mps2: np.ndarray
    potentials_m2_s2: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedPointMassSystem:
    """External/external terms shared by all corrections at one epoch."""

    integrated_count: int
    external_accelerations_mps2: np.ndarray
    external_potentials_m2_s2: np.ndarray


def _newtonian_acceleration_and_potential(positions, gm):
    r = np.ascontiguousarray(positions, dtype=float)
    mu = np.ascontiguousarray(gm, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3 or mu.shape != (len(r),):
        raise ValueError("Expected positions (N,3) and GM (N,).")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("Positions must be finite and GM positive.")
    return point_mass_symmetric(r, mu)


def prepare_point_mass_system(positions, gm, *, integrated_count):
    """Precompute the prescribed-body Newtonian subsystem for one epoch."""
    r = np.asarray(positions, dtype=float)
    mu = np.asarray(gm, dtype=float)
    count = int(integrated_count)
    if count < 0 or count > len(r):
        raise ValueError("integrated_count is outside the body array")
    external_acceleration, external_potential = _newtonian_acceleration_and_potential(r[count:], mu[count:])
    return PreparedPointMassSystem(count, external_acceleration, external_potential)


def evaluate_point_mass_system(positions, gm, prepared=None):
    """Evaluate all Newtonian accelerations and potentials, reusing fixed source terms."""
    r = np.asarray(positions, dtype=float)
    mu = np.asarray(gm, dtype=float)
    if prepared is None:
        acceleration, potential = _newtonian_acceleration_and_potential(r, mu)
        return NewtonianPointMassEvaluation(acceleration, potential)
    if not isinstance(prepared, PreparedPointMassSystem):
        raise TypeError("prepared must be a PreparedPointMassSystem")
    integrated_count = prepared.integrated_count
    if r.ndim != 2 or r.shape[1] != 3 or mu.shape != (len(r),):
        raise ValueError("Expected positions (N,3) and GM (N,).")
    if prepared.external_accelerations_mps2.shape != (len(r) - integrated_count, 3):
        raise ValueError("Prepared point-mass system does not match the body array")

    acceleration = np.empty_like(r)
    potential = np.empty(len(r), dtype=float)
    if integrated_count:
        targets = r[:integrated_count]
        delta = r[None, :, :] - targets[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        distance[np.arange(integrated_count), np.arange(integrated_count)] = np.inf
        if np.any(distance == 0):
            raise ValueError("Distinct bodies have coincident positions.")
        inverse_distance = 1.0 / distance
        acceleration[:integrated_count] = np.sum(
            delta * (mu[None, :] * inverse_distance**3)[:, :, None], axis=1
        )
        potential[:integrated_count] = inverse_distance @ mu

    external_count = len(r) - integrated_count
    if external_count:
        acceleration[integrated_count:] = prepared.external_accelerations_mps2
        potential[integrated_count:] = prepared.external_potentials_m2_s2
        if integrated_count:
            delta = r[:integrated_count][None, :, :] - r[integrated_count:, None, :]
            distance = np.linalg.norm(delta, axis=2)
            if np.any(distance == 0):
                raise ValueError("Distinct bodies have coincident positions.")
            inverse_distance = 1.0 / distance
            acceleration[integrated_count:] += np.sum(
                delta * (mu[:integrated_count][None, :] * inverse_distance**3)[:, :, None], axis=1
            )
            potential[integrated_count:] += inverse_distance @ mu[:integrated_count]
    return NewtonianPointMassEvaluation(acceleration, potential)


def eih_acceleration(positions, velocities, gm, newtonian, target_indices):
    """Evaluate Eq. 27 EIH corrections only for the requested target rows."""
    r, v, mu = (np.asarray(value, dtype=float) for value in (positions, velocities, gm))
    targets = np.asarray(target_indices, dtype=int)
    if r.ndim != 2 or r.shape[1] != 3 or v.shape != r.shape or mu.shape != (len(r),):
        raise ValueError("Expected positions/velocities (N,3) and GM (N,).")
    if not isinstance(newtonian, NewtonianPointMassEvaluation):
        raise TypeError("newtonian must be a NewtonianPointMassEvaluation")
    if np.any(targets < 0) or np.any(targets >= len(r)):
        raise ValueError("target_indices contains an invalid body index")
    if len(targets) == 0:
        return np.zeros_like(r)

    delta = r[None, :, :] - r[targets, None, :]
    distance = np.linalg.norm(delta, axis=2)
    distance[np.arange(len(targets)), targets] = np.inf
    if np.any(distance == 0):
        raise ValueError("Distinct bodies have coincident positions.")
    inverse_distance = 1.0 / distance
    weight = mu[None, :] * inverse_distance**3
    vi = v[targets, None, :]
    vj = v[None, :, :]
    source_acceleration = newtonian.accelerations_mps2[None, :, :]
    dot = lambda left, right: np.sum(left * right, axis=-1)
    factor = (
        -4 * newtonian.potentials_m2_s2[targets, None]
        - newtonian.potentials_m2_s2[None, :]
        + dot(vi, vi)
        + 2 * dot(vj, vj)
        - 4 * dot(vi, vj)
        - 1.5 * (dot(delta, vj) * inverse_distance) ** 2
        + 0.5 * dot(delta, source_acceleration)
    )
    target_correction = (
        np.sum(
            weight[:, :, None]
            * (delta * factor[:, :, None] + dot(delta, 4 * vi - 3 * vj)[:, :, None] * (vj - vi)),
            axis=1,
        )
        + 3.5 * (inverse_distance * mu[None, :]) @ newtonian.accelerations_mps2
    ) / C**2
    correction = np.zeros_like(r)
    correction[targets] = target_correction
    return correction


def point_mass_acceleration(positions, gm):
    """Evaluate the complete Newtonian point-mass auxiliary system."""
    return evaluate_point_mass_system(positions, gm).accelerations_mps2
