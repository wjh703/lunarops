"""Delayed whole-Moon inertia, its derivative, and the shared degree-2 field.

:class:`LunarDeformation` follows the integrated attitude; :class:`LunarDegree2GravityModel`
evaluates the same response along a prescribed orbit and attitude.

All tensors use whole-Moon M R² normalization. The fluid core is constant;
the tidal and centrifugal deformations belong to the mantle.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lunarops.classes.time import require_tdb_epoch

from ._vector import cross3
from .gravity import GravityCoefficients

# Coefficient order is (C20, C21, C22, S20, S21, S22), and inertia order is
# (Ixx, Iyy, Izz, Ixy, Ixz, Iyz).
_DEGREE2_GRAVITY_FROM_INERTIA = np.array(
    (
        (1 / (2 * np.sqrt(5)), 1 / (2 * np.sqrt(5)), -1 / np.sqrt(5), 0, 0, 0),
        (0, 0, 0, 0, -np.sqrt(3 / 5), 0),
        (-np.sqrt(12 / 5) / 4, np.sqrt(12 / 5) / 4, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, -np.sqrt(3 / 5)),
        (0, 0, 0, -np.sqrt(12 / 5) / 2, 0, 0),
    ),
    dtype=float,
)
_DEGREE2_GRAVITY_FROM_INERTIA.setflags(write=False)


def _inertia_components(tensor: np.ndarray) -> np.ndarray:
    return np.asarray(
        (tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]),
        dtype=float,
    )


def _validated_inertia(inertia) -> np.ndarray:
    tensor = np.asarray(inertia, dtype=float)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("Inertia must be a finite 3x3 tensor.")
    if not np.allclose(tensor, tensor.T, rtol=0, atol=1e-15):
        raise ValueError("Inertia must be symmetric.")
    return tensor


def degree2_gravity_coefficients_from_inertia(
    inertia, *, gravitational_parameter_m3_s2, radius_m,
    c21_static=0.0, s21_static=0.0, s22_static=0.0,
) -> GravityCoefficients:
    """Inertia-derived gravity plus explicit unnormalized static offsets."""
    coefficients = np.zeros((2, 3, 3), dtype=float)
    coefficients[0, 0, 0] = 1.0
    coefficients[:, 2, :3] = _degree2_coefficients_from_inertia(
        _validated_inertia(inertia), c21_static=c21_static, s21_static=s21_static, s22_static=s22_static,
    )
    return GravityCoefficients(coefficients, float(gravitational_parameter_m3_s2), float(radius_m))


def _degree2_coefficients_from_inertia(
    inertia, *, c21_static=0.0, s21_static=0.0, s22_static=0.0,
) -> np.ndarray:
    """Fully normalized degree 2; Pavlov Eq. (16) static offsets do not alter mechanical inertia."""
    coefficients = (
        _DEGREE2_GRAVITY_FROM_INERTIA @ _inertia_components(inertia)
    ).reshape(2, 3)
    coefficients[0, 1] += c21_static * np.sqrt(3 / 5)
    coefficients[1, 1] += s21_static * np.sqrt(3 / 5)
    coefficients[1, 2] += s22_static * np.sqrt(12 / 5)
    return coefficients



from .rotation import quaternion_matrix


@dataclass(frozen=True, slots=True)
class LunarDeformationParameters:
    love_k2: float
    delay_s: float
    mean_motion_rad_s: float
    include_tidal: bool = True
    include_rotational: bool = True
    # Unnormalized C21^(0), S21^(0), S22^(0) in Pavlov Eq. (16).
    c21_static: float = 0.0
    s21_static: float = 0.0
    s22_static: float = 0.0



@dataclass(frozen=True, slots=True)
class LunarInertiaEvaluation:
    total: np.ndarray
    derivative: np.ndarray
    tidal: np.ndarray
    rotational: np.ndarray
    # Fully normalized degree-2 (C, S) x orders 0..2 of the shared Moon field.
    degree2: np.ndarray | None = None


def deformation_tensors(r, r_dot, omega, omega_dot, *, parameters, radius_m, earth_mu, moon_mu):
    """DE430 Eqs. 40–42 / DE440 Eqs. 47–55 and analytic time derivatives.

    Position, its derivative and both angular quantities are PA components
    at the SAME delayed epoch. No differencing of attitude or RHS recursion.
    """
    p = parameters
    eye = np.eye(3)
    distance = np.linalg.norm(r)
    rr = r @ r_dot
    Q = np.outer(r, r) - distance**2 * eye / 3
    Q_dot = np.outer(r_dot, r) + np.outer(r, r_dot) - 2 * rr * eye / 3
    a = p.love_k2 * earth_mu / moon_mu * radius_m**3
    tidal = -a * Q / distance**5
    tidal_dot = a * (5 * rr * Q / distance**7 - Q_dot / distance**5)
    b = p.love_k2 * radius_m**3 / (3 * moon_mu)
    spin = np.outer(omega, omega) - eye * (omega @ omega - p.mean_motion_rad_s**2) / 3
    spin[2, 2] -= p.mean_motion_rad_s**2
    rotational = b * spin
    rotational_dot = b * (
        np.outer(omega_dot, omega) + np.outer(omega, omega_dot) - 2 * (omega @ omega_dot) * eye / 3
    )
    if not p.include_tidal:
        tidal.fill(0)
        tidal_dot.fill(0)
    if not p.include_rotational:
        rotational.fill(0)
        rotational_dot.fill(0)
    return tidal, rotational, tidal_dot + rotational_dot


class LunarDeformation:
    """One delayed evaluation supplies mantle dynamics and lunar gravity.

    ``field`` is the Moon field shared by MoonFigureForce and LunarFigureTorque. Its
    degree 2 is replaced, per evaluation, by the delayed inertia; it is never added
    to the static degree 2 and the field object is never modified.
    """

    def __init__(self, parameters, total_inertia, radius_m, earth_mu, moon_mu, field):
        self.parameters = parameters
        self.undistorted_inertia = total_inertia
        self.radius_m = radius_m
        self.earth_mu, self.moon_mu = earth_mu, moon_mu
        self.field = field

    def evaluate(self, epoch_tdb, history, *, rigid=False):
        """Delayed inertia and the matching degree-2 gravity from accepted history."""
        if rigid:
            total = self.undistorted_inertia
            tidal, rotational, derivative = (np.zeros((3, 3)) for _ in range(3))
        else:
            state, state_dot = history.integration_sample(epoch_tdb.shifted(-self.parameters.delay_s))
            R = quaternion_matrix(state[6:10])
            omega = state[10:13]
            r = -R.T @ state[:3]
            r_dot = -R.T @ state[3:6] - cross3(omega, r)
            tidal, rotational, derivative = deformation_tensors(
                r, r_dot, omega, state_dot[10:13],
                parameters=self.parameters, radius_m=self.radius_m,
                earth_mu=self.earth_mu, moon_mu=self.moon_mu,
            )
            total = self.undistorted_inertia + tidal + rotational
        p = self.parameters
        degree2 = _degree2_coefficients_from_inertia(
            total, c21_static=p.c21_static, s21_static=p.s21_static, s22_static=p.s22_static,
        )
        return LunarInertiaEvaluation(total, derivative, tidal, rotational, degree2)


@dataclass(frozen=True, slots=True)
class LunarDegree2GravityParameters:
    """Parameters for the delayed lunar degree-2 gravity response."""

    # DE440 BSP comment-area constants, not the mean observed quadrupole.
    j2_undistorted: float = 2.0321436013500001e-4
    beta: float = 6.3122040737914229e-4
    gamma: float = 2.2778898477433167e-4
    love_k2: float = 0.02419
    tidal_delay_days: float = 0.14519388258636395
    radius_m: float = 1738000.0
    mean_motion_rad_s: float = 2.6616995e-6
    include_tidal_deformation: bool = True
    include_rotational_deformation: bool = True
    # Unnormalized static offsets in Pavlov et al. (2016), Eq. (16).
    c21_static: float = 0.0
    s21_static: float = 0.0
    s22_static: float = 0.0

    def __post_init__(self) -> None:
        if not (0 < self.gamma < self.beta < 1) or self.j2_undistorted <= 0:
            raise ValueError("Lunar deformation requires 0 < gamma < beta < 1 and positive J2.")

    def undistorted_inertia(self) -> np.ndarray:
        """Return the normalized undeformed lunar inertia tensor."""
        b, g = self.beta, self.gamma
        factor = 2 * self.j2_undistorted / (2 * b - g + b * g)
        return np.diag([factor * (1 - b * g), factor * (1 + g), factor * (1 + b)])


class LunarDegree2GravityModel:
    """Compute delayed lunar degree-2 gravity from prescribed orbit and attitude."""

    def __init__(
        self,
        earth_gravitational_parameter_m3_s2,
        moon_gravitational_parameter_m3_s2,
        earth_minus_moon_position_provider: Callable,
        attitude_provider: Callable,
        *,
        angular_velocity_provider: Callable,
        parameters: LunarDegree2GravityParameters | None = None,
    ) -> None:
        self.earth_gravitational_parameter_m3_s2 = float(earth_gravitational_parameter_m3_s2)
        self.moon_gravitational_parameter_m3_s2 = float(moon_gravitational_parameter_m3_s2)
        self.earth_minus_moon_position_provider = earth_minus_moon_position_provider
        self.attitude_provider = attitude_provider
        self.angular_velocity_provider = angular_velocity_provider
        self.parameters = parameters if parameters is not None else LunarDegree2GravityParameters()
        self._response = LunarDeformationParameters(
            self.parameters.love_k2, self.parameters.tidal_delay_days * 86400,
            self.parameters.mean_motion_rad_s,
            self.parameters.include_tidal_deformation, self.parameters.include_rotational_deformation,
            self.parameters.c21_static, self.parameters.s21_static, self.parameters.s22_static,
        )

    def inertia(self, epoch, earth_minus_moon_position_provider=None) -> np.ndarray:
        """Whole-Moon inertia of the delayed prescribed orbit and attitude."""
        require_tdb_epoch(epoch)
        p = self.parameters
        delayed = epoch.shifted(-p.tidal_delay_days * 86400)
        provider = earth_minus_moon_position_provider or self.earth_minus_moon_position_provider
        r = np.asarray(self.attitude_provider(delayed), dtype=float).T @ np.asarray(provider(delayed), dtype=float)
        tide, spin, _ = deformation_tensors(
            r, np.zeros(3), self.angular_velocity_provider(delayed), np.zeros(3), parameters=self._response,
            radius_m=p.radius_m, earth_mu=self.earth_gravitational_parameter_m3_s2,
            moon_mu=self.moon_gravitational_parameter_m3_s2,
        )
        return p.undistorted_inertia() + tide + spin

    def degree2_coefficients(self, epoch, earth_minus_moon_position_provider=None) -> np.ndarray:
        """Fully normalized degree 2 of the delayed Moon; higher degrees stay with the field."""
        p = self.parameters
        return _degree2_coefficients_from_inertia(
            self.inertia(epoch, earth_minus_moon_position_provider),
            c21_static=p.c21_static, s21_static=p.s21_static, s22_static=p.s22_static,
        )
