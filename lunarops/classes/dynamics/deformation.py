"""Delayed whole-Moon inertia, its derivative, and the shared degree-2 field.

All tensors use whole-Moon M R² normalization. The fluid core is constant;
the tidal and centrifugal deformations belong to the mantle.
"""

from dataclasses import dataclass

import numpy as np

from .inertia import _degree2_coefficients_from_inertia
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

    def __post_init__(self):
        if not np.isfinite(self.delay_s) or self.delay_s <= 0:
            raise ValueError("Delayed lunar deformation requires a positive finite delay")
        if not np.isfinite(self.love_k2) or self.love_k2 < 0:
            raise ValueError("Lunar Love k2 must be nonnegative and finite")
        if not np.isfinite(self.mean_motion_rad_s) or self.mean_motion_rad_s <= 0:
            raise ValueError("Lunar mean motion must be positive and finite (rad/s)")


@dataclass(frozen=True, slots=True)
class LunarInertiaEvaluation:
    total: np.ndarray
    derivative: np.ndarray
    tidal: np.ndarray
    rotational: np.ndarray


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

    The field is the SAME object consumed by FigureForce/LunarFigureTorque.
    Replace its degree 2; never add a second static degree-2 field.
    """

    def __init__(self, parameters, total_inertia, radius_m, earth_mu, moon_mu, field):
        self.parameters = parameters
        self.undistorted_inertia = total_inertia
        self.radius_m = radius_m
        self.earth_mu, self.moon_mu = earth_mu, moon_mu
        self.field = field

    def evaluate(self, inputs, *, rigid=False):
        if rigid:
            total = self.undistorted_inertia
            tidal, rotational, derivative = (np.zeros((3, 3)) for _ in range(3))
        else:
            epoch = inputs.epoch_data.epoch_tdb.shifted(-self.parameters.delay_s)
            state, state_dot = inputs.history.integration_sample(epoch)
            R = quaternion_matrix(state[6:10])
            omega = state[10:13]
            r = -R.T @ state[:3]
            r_dot = -R.T @ state[3:6] - np.cross(omega, r)
            tidal, rotational, derivative = deformation_tensors(
                r, r_dot, omega, state_dot[10:13],
                parameters=self.parameters, radius_m=self.radius_m,
                earth_mu=self.earth_mu, moon_mu=self.moon_mu,
            )
            total = self.undistorted_inertia + tidal + rotational
        p = self.parameters
        self.field.update_degree2_coefficients(_degree2_coefficients_from_inertia(
            total, c21_static=p.c21_static, s21_static=p.s21_static, s22_static=p.s22_static,
        ))
        return LunarInertiaEvaluation(total, derivative, tidal, rotational)
