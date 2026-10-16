"""Hamilton PA-to-inertial kinematics and rigid mantle/fluid-core dynamics.

Inertias are divided by the whole Moon's M R². Angular velocities are
mantle-frame components in rad per TDB second; torques have units s^-2.
The optional anelastic model supplies the mantle tensor and its derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lunarops.base.array_validation import finite_array
from lunarops.base.constants import C

if TYPE_CHECKING:
    from .deformation import LunarInertiaEvaluation


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


def quaternion_derivative(quaternion, angular_velocity) -> np.ndarray:
    """q_dot = 1/2 q ⊗ (0, omega_PA), without state projection."""
    q = np.asarray(quaternion)
    w = np.asarray(angular_velocity)
    return 0.5 * np.concatenate(([-q[1:] @ w], q[0] * w + np.cross(q[1:], w)))


@dataclass(frozen=True, slots=True)
class LunarRotationState:
    quaternion: np.ndarray
    angular_velocity_pa_rad_s: np.ndarray
    core_angular_velocity_pa_rad_s: np.ndarray | None = None

    def __post_init__(self):
        for key, shape in (("quaternion", (4,)), ("angular_velocity_pa_rad_s", (3,))):
            object.__setattr__(
                self, key, finite_array(getattr(self, key), shape=shape, name=key, copy=True, readonly=True)
            )
        if not np.isclose(np.linalg.norm(self.quaternion), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Initial attitude quaternion must have unit norm")
        if self.core_angular_velocity_pa_rad_s is not None:
            object.__setattr__(
                self,
                "core_angular_velocity_pa_rad_s",
                finite_array(
                    self.core_angular_velocity_pa_rad_s,
                    shape=(3,),
                    name="core_angular_velocity_pa_rad_s",
                    copy=True,
                    readonly=True,
                ),
            )

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
    pressure_spin: str

    def __post_init__(self):
        if not 0 < self.polar_moment_fraction < 1 or not 0 <= self.flattening < 1:
            raise ValueError("Core moment fraction and flattening must define a positive inertia")
        if not np.isfinite(self.viscosity_per_s) or self.viscosity_per_s < 0:
            raise ValueError("Core viscosity must be nonnegative and finite (s^-1, normalized by whole-Moon M R²)")
        if self.pressure_spin not in ("core", "mantle"):
            raise ValueError("pressure_spin must explicitly select 'core' (DE430) or 'mantle' (DE440 printed Eq. 58)")


@dataclass(frozen=True, slots=True)
class RotationEvaluation:
    derivative: np.ndarray
    torques: dict[str, np.ndarray]
    viscous_torque: np.ndarray
    pressure_torque: np.ndarray
    geodetic_rate_pa_rad_s: np.ndarray
    inertia: LunarInertiaEvaluation | None


class LunarRotationDynamics:
    """Rigid or deforming mantle and optional axisymmetric fluid core.

    Both core and mantle angular velocities are expressed in the mantle
    frame. Core orientation angles do not enter this model's feedback.
    """

    def __init__(self, total_inertia, radius_m, torque_group, *, core=None, geodetic_precession=False, deformation=None):
        self.total_inertia = finite_array(total_inertia, shape=(3, 3), name="total_inertia", copy=True, readonly=True)
        if not np.isfinite(radius_m) or radius_m <= 0:
            raise ValueError("Lunar reference radius must be positive and finite")
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
        self._core_diagonal = np.diag(self.core_inertia).copy()
        self._core_diagonal.setflags(write=False)
        self._mantle_diagonal = None
        if np.allclose(self.mantle_inertia, np.diag(np.diag(self.mantle_inertia)), rtol=0.0, atol=1e-15):
            diagonal = np.diag(self.mantle_inertia).copy()
            diagonal.setflags(write=False)
            self._mantle_diagonal = diagonal
        self.core_inertia.setflags(write=False)
        self.mantle_inertia.setflags(write=False)

    @property
    def state_dimension(self):
        return 7 if self.core is None else 10

    def evaluate(self, inputs, state) -> RotationEvaluation:
        q, wm = state[:4], state[4:7]
        mantle = self.mantle_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total - self.core_inertia
        inertia_dot = np.zeros((3, 3)) if inputs.lunar_inertia is None else inputs.lunar_inertia.derivative
        total, terms = self.torque_group.evaluate(inputs)
        nu = self.geodetic_rate(inputs) if self.geodetic_precession else np.zeros(3)
        viscous, pressure = np.zeros(3), np.zeros(3)
        if self.core is not None:
            wc = state[7:10]
            viscous = self.core.viscosity_per_s * (wc - wm)
            spin = wc[2] if self.core.pressure_spin == "core" else wm[2]
            pressure = (self.core_inertia[2, 2] - self.core_inertia[0, 0]) * spin * np.array([-wc[1], wc[0], 0.0])
        cmb = viscous + pressure
        mantle_rhs = total + cmb - np.cross(wm - nu, mantle @ wm) - inertia_dot @ wm
        if inputs.lunar_inertia is None and self._mantle_diagonal is not None:
            mantle_dot = mantle_rhs / self._mantle_diagonal
        else:
            # Deformation makes the mantle tensor time-varying; retain a
            # stable general solve for that case.
            mantle_dot = np.linalg.solve(mantle, mantle_rhs)
        parts = [quaternion_derivative(q, wm), mantle_dot]
        if self.core is not None:
            core_rhs = -cmb - np.cross(wm - nu, self.core_inertia @ wc)
            parts.append(core_rhs / self._core_diagonal)
        return RotationEvaluation(np.concatenate(parts), terms, viscous, pressure, nu, inputs.lunar_inertia)

    @staticmethod
    def geodetic_rate(inputs) -> np.ndarray:
        """DE440 Eq. 44, gamma_PPN=1, SI/TDB and current candidate PA."""
        earth, moon, sun = (inputs.body_indices[name] for name in ("EARTH", "MOON", "SUN"))
        r_em = inputs.positions_m[moon] - inputs.positions_m[earth]
        v_em = inputs.velocities_mps[moon] - inputs.velocities_mps[earth]
        r_sm = inputs.positions_m[moon] - inputs.positions_m[sun]
        mu = inputs.gravitational_parameters_m3_s2
        field = mu[earth] * r_em / np.linalg.norm(r_em) ** 3 + mu[sun] * r_sm / np.linalg.norm(r_sm) ** 3
        return inputs.moon_fixed2inertial_matrix.T @ (1.5 / C**2 * np.cross(v_em, field))
