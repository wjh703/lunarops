"""Lunar attitude kinematics and mantle/fluid-core rotational dynamics.

The lunar attitude is a scalar-first Hamilton quaternion that rotates Moon
principal-axis (PA) components into inertial (ICRF-aligned) components.
Inertia tensors are normalised by the whole-Moon ``M R²``; angular velocities
are PA (mantle-frame) components in rad s⁻¹ of TDB; torques have units s⁻².
The optional deformation model supplies the delayed mantle tensor and its
time derivative.

References
----------
Folkner et al. (2014), Eqs. 33–45; Park et al. (2021), Eqs. 42–58.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quaternion_matrix(quaternion) -> np.ndarray:
    """Return the rotation matrix of a scalar-first Hamilton quaternion.

    The matrix is the active rotation ``R`` with ``v_inertial = R @ v_pa``.
    The quaternion is normalised for this evaluation only; the integrated
    state itself is never renormalised.

    Parameters
    ----------
    quaternion : array_like
        Quaternion ``(a, x, y, z)`` with scalar part first, shape ``(4,)``.
        Any nonzero norm is accepted.

    Returns
    -------
    numpy.ndarray
        Orthonormal rotation matrix of shape ``(3, 3)``.
    """
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
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
    """Initial rotational state of the Moon.

    Attributes
    ----------
    quaternion : numpy.ndarray
        Scalar-first Hamilton quaternion from PA to inertial axes, shape
        ``(4,)``.
    angular_velocity_pa_rad_s : numpy.ndarray
        Mantle angular velocity in PA components, rad s⁻¹, shape ``(3,)``.
    core_angular_velocity_pa_rad_s : numpy.ndarray or None, default None
        Fluid-core angular velocity in PA (mantle-frame) components,
        rad s⁻¹, shape ``(3,)``; ``None`` without a core model.
    """

    quaternion: np.ndarray
    angular_velocity_pa_rad_s: np.ndarray
    core_angular_velocity_pa_rad_s: np.ndarray | None = None

    def vector(self) -> np.ndarray:
        """Return the rotational part of the integration state.

        Returns
        -------
        numpy.ndarray
            ``[q (4), ω_mantle (3)]`` of shape ``(7,)``, or
            ``[q (4), ω_mantle (3), ω_core (3)]`` of shape ``(10,)`` with a
            core.
        """
        parts = [self.quaternion, self.angular_velocity_pa_rad_s]
        if self.core_angular_velocity_pa_rad_s is not None:
            parts.append(self.core_angular_velocity_pa_rad_s)
        return np.concatenate(parts)


@dataclass(frozen=True, slots=True)
class LunarCoreParameters:
    """Axisymmetric fluid-core parameters.

    Attributes
    ----------
    polar_moment_fraction : float
        Core polar moment ``C_c`` as a fraction of the whole-Moon polar moment
        ``C`` (dimensionless).
    flattening : float
        Core flattening ``f_c``; the core tensor is
        ``C_c · diag(1 - f_c, 1 - f_c, 1)`` (dimensionless).
    viscosity_per_s : float
        Core-mantle boundary friction coefficient ``k_v`` normalised by the
        whole-Moon ``M R²``, in s⁻¹. The viscous torque on the mantle is
        ``k_v (ω_core - ω_mantle)``.

    References
    ----------
    Folkner et al. (2014), Eq. 45; Park et al. (2021), Eq. 58.
    """

    polar_moment_fraction: float
    flattening: float
    viscosity_per_s: float


class LunarRotationDynamics:
    """Rotational dynamics of the lunar mantle and an optional fluid core.

    The mantle is rigid or follows the delayed tidal and rotational
    deformation of :class:`~lunarops.classes.dynamics.deformation.LunarDeformation`.
    Core and mantle angular velocities are both expressed in the mantle (PA)
    frame. Core orientation angles are not integrated.

    Parameters
    ----------
    total_inertia : array_like
        Undeformed whole-Moon inertia tensor in PA axes, shape ``(3, 3)``,
        normalised by ``M R²``.
    radius_m : float
        Lunar reference radius ``R`` in m.
    torque_group : LunarTorqueGroup
        External torques acting on the mantle.
    core : LunarCoreParameters or None, optional
        Fluid-core parameters; ``None`` for a single rigid body.
    geodetic_precession : bool, default False
        Include the de Sitter (geodetic) precession of the PA frame due to
        the Earth and the Sun.
    deformation : LunarDeformation or None, optional
        Delayed tidal and rotational deformation of the mantle.

    Attributes
    ----------
    core_inertia : numpy.ndarray
        Core inertia tensor in PA axes, shape ``(3, 3)``; zeros without a core.
    mantle_inertia : numpy.ndarray
        Undeformed mantle inertia ``total_inertia - core_inertia``,
        shape ``(3, 3)``.

    Raises
    ------
    ValueError
        If the mantle inertia is not positive definite.

    References
    ----------
    Park et al. (2021), Eqs. 43–45 (mantle, geodetic precession) and Eq. 58
    (core).
    """

    def __init__(self, total_inertia, radius_m, torque_group, *, core=None, geodetic_precession=False, deformation=None):
        self.total_inertia = np.asarray(total_inertia, float)
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
        """int: Rotational state length, 7 without a core and 10 with one."""
        return 7 if self.core is None else 10
