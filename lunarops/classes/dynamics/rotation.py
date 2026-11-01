"""Hamilton PA-to-inertial kinematics and rigid mantle/fluid-core dynamics.

Inertias are divided by the whole Moon's M R². Angular velocities are
mantle-frame components in rad per TDB second; torques have units s^-2.
The optional anelastic model supplies the mantle tensor and its derivative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.base.array_validation import finite_array


def quaternion_matrix(quaternion) -> np.ndarray:
    """Active rotation, Hamilton quaternion with scalar first.

    Normalize only the rotation evaluation, never the integrator's input.
    """
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("A finite nonzero attitude quaternion is required")
    a, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - a * z), 2 * (x * z + a * y)],
            [2 * (x * y + a * z), 1 - 2 * (x * x + z * z), 2 * (y * z - a * x)],
            [2 * (x * z - a * y), 2 * (y * z + a * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True, slots=True)
class LunarRotationState:
    quaternion: np.ndarray
    angular_velocity_pa_rad_s: np.ndarray
    core_angular_velocity_pa_rad_s: np.ndarray | None = None

    def vector(self) -> np.ndarray:
        parts = [self.quaternion, self.angular_velocity_pa_rad_s]
        if self.core_angular_velocity_pa_rad_s is not None:
            parts.append(self.core_angular_velocity_pa_rad_s)
        return np.concatenate(parts)


@dataclass(frozen=True, slots=True)
class LunarCoreParameters:
    polar_moment_fraction: float
    flattening: float
    viscosity_per_s: float


class LunarRotationDynamics:
    """Rigid or deforming mantle and optional axisymmetric fluid core.

    Both core and mantle angular velocities are expressed in the mantle
    frame. Core orientation angles do not enter this model's feedback.
    """

    def __init__(self, total_inertia, radius_m, torque_group, *, core=None, geodetic_precession=False, deformation=None):
        self.total_inertia = finite_array(total_inertia, shape=(3, 3), name="total_inertia", copy=True, readonly=True)
        self.radius_m = float(radius_m)
        self.core = core
        self.torque_group = torque_group
        self.geodetic_precession = geodetic_precession
        self.deformation = deformation
        self.core_inertia = (
            np.zeros((3, 3))
            if core is None
            else (
                core.polar_moment_fraction
                * self.total_inertia[2, 2]
                * np.diag([1 - core.flattening, 1 - core.flattening, 1])
            )
        )
        self.mantle_inertia = self.total_inertia - self.core_inertia
        if np.linalg.eigvalsh(self.mantle_inertia)[0] <= 0:
            raise ValueError("Mantle inertia must be positive definite after subtracting the core")

    @property
    def state_dimension(self):
        return 7 if self.core is None else 10
