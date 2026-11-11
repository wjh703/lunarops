"""Lunar attitude kinematics and mantle/fluid-core rotational dynamics.

The lunar attitude is the set of 3-1-3 Euler angles (φ, θ, ψ) that rotates Moon
principal-axis (PA) components into inertial (ICRF-aligned) components.
Inertia tensors are normalised by the whole-Moon ``M R²``; angular velocities
are PA (mantle-frame) components in rad s⁻¹ of TDB; torques have units s⁻².
The optional deformation model supplies the delayed mantle tensor and its
time derivative.

References
----------
Folkner et al. (2014), Eqs. 13–14 and 33–45; Park et al. (2021), Eqs. 42–58.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def euler_matrix(angles) -> np.ndarray:
    """Return the rotation matrix of 3-1-3 Euler angles.

    The matrix is the active rotation ``R = Rz(φ) Rx(θ) Rz(ψ)`` with
    ``v_inertial = R @ v_pa``.

    Parameters
    ----------
    angles : array_like
        Angles ``(φ, θ, ψ)`` in rad, shape ``(3,)``.

    Returns
    -------
    numpy.ndarray
        Orthonormal rotation matrix of shape ``(3, 3)``.
    """
    phi, theta, psi = np.asarray(angles, dtype=float)
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cf * cp - sf * ct * sp, -cf * sp - sf * ct * cp, sf * st],
            [sf * cp + cf * ct * sp, -sf * sp + cf * ct * cp, -cf * st],
            [st * sp, st * cp, ct],
        ]
    )


@dataclass(frozen=True, slots=True)
class LunarRotationState:
    """Initial rotational state of the Moon.

    Attributes
    ----------
    euler_angles_rad : numpy.ndarray
        3-1-3 Euler angles ``(φ, θ, ψ)`` from PA to inertial axes, rad, shape
        ``(3,)``. ``sin θ`` must be nonzero.
    angular_velocity_pa_rad_s : numpy.ndarray
        Mantle angular velocity in PA components, rad s⁻¹, shape ``(3,)``.
    core_angular_velocity_pa_rad_s : numpy.ndarray or None, default None
        Fluid-core angular velocity in PA (mantle-frame) components,
        rad s⁻¹, shape ``(3,)``; ``None`` without a core model.
    """

    euler_angles_rad: np.ndarray
    angular_velocity_pa_rad_s: np.ndarray
    core_angular_velocity_pa_rad_s: np.ndarray | None = None

    def vector(self) -> np.ndarray:
        """Return the rotational part of the integration state.

        Returns
        -------
        numpy.ndarray
            ``[φ, θ, ψ, ω_mantle (3)]`` of shape ``(6,)``, or
            ``[φ, θ, ψ, ω_mantle (3), ω_core (3)]`` of shape ``(9,)`` with a
            core.
        """
        parts = [self.euler_angles_rad, self.angular_velocity_pa_rad_s]
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

    def __init__(
        self, total_inertia, radius_m, torque_group, *, core=None, geodetic_precession=False, deformation=None
    ):
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
        """int: Rotational state length, 6 without a core and 9 with one."""
        return 6 if self.core is None else 9
