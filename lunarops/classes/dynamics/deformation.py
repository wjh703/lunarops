"""Delayed tidal and rotational deformation of the Moon.

The Moon's inertia tensor responds, with a constant time delay, to the tide
raised by the Earth and to its own spin. The same delayed tensor defines the
degree-2 lunar gravity field.

- :class:`LunarDeformation` evaluates the response along the integrated
  orbit and attitude and feeds both the rotational dynamics and the lunar
  figure force.
- :class:`LunarDegree2GravityModel` evaluates it along a prescribed
  (ephemeris) orbit and attitude for orbit-only propagation.

All tensors are in Moon principal-axis (PA) components and normalised by the
whole-Moon ``M R²``. The fluid core is undeformed; the tidal and rotational
deformations belong to the mantle.

References
----------
Folkner et al. (2014), Eqs. 40–42; Park et al. (2021), Eqs. 47–55;
Pavlov et al. (2016), Eq. 16 (static degree-2 offsets).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lunarops.classes.time import require_tdb_epoch

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


def _inertia_components(tensor: np.ndarray) -> np.ndarray:
    return np.asarray(
        (tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]),
        dtype=float,
    )


def _degree2_coefficients_from_inertia(
    inertia, *, c21_static=0.0, s21_static=0.0, s22_static=0.0,
) -> np.ndarray:
    """Return fully normalised degree-2 coefficients of an inertia tensor.

    The static offsets of Pavlov et al. (2016), Eq. 16, are added to the
    gravity coefficients only; they do not alter the mechanical inertia.
    The result has shape ``(2, 3)``: rows C and S, columns orders 0–2.
    """
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
    """Parameters of the delayed lunar deformation response.

    Attributes
    ----------
    love_k2 : float
        Lunar potential Love number k2 (dimensionless).
    delay_s : float
        Constant response delay τ in TDB seconds.
    mean_motion_rad_s : float
        Lunar mean motion n in rad s⁻¹; the spin deformation is referred to
        uniform rotation at this rate about the PA z axis.
    include_tidal : bool, default True
        Include the Earth-raised tidal deformation.
    include_rotational : bool, default True
        Include the spin (centrifugal) deformation.
    c21_static, s21_static, s22_static : float, default 0.0
        Unnormalised static offsets C21⁽⁰⁾, S21⁽⁰⁾, S22⁽⁰⁾ added to the gravity
        coefficients only (Pavlov et al. 2016, Eq. 16).
    """

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
    """Delayed whole-Moon inertia evaluated for one integration step.

    All tensors are in PA axes, normalised by ``M R²``, shape ``(3, 3)``.

    Attributes
    ----------
    total : numpy.ndarray
        Undeformed tensor plus tidal and rotational parts.
    derivative : numpy.ndarray
        Time derivative of ``total`` in s⁻¹ at the delayed epoch.
    tidal : numpy.ndarray
        Tidal part of the deformation.
    rotational : numpy.ndarray
        Rotational (spin) part of the deformation.
    degree2 : numpy.ndarray or None, default None
        Fully normalised degree-2 coefficients of the shared Moon field,
        shape ``(2, 3)`` (rows C and S, columns orders 0–2).
    """

    total: np.ndarray
    derivative: np.ndarray
    tidal: np.ndarray
    rotational: np.ndarray
    # Fully normalized degree-2 (C, S) x orders 0..2 of the shared Moon field.
    degree2: np.ndarray | None = None


def deformation_tensors(r, r_dot, omega, omega_dot, *, parameters, radius_m, earth_mu, moon_mu):
    """Return the tidal and rotational inertia deformations and their rate.

    The derivative is analytic; no attitude differencing or nested
    right-hand-side evaluation is used.

    Parameters
    ----------
    r : numpy.ndarray
        Moon-to-Earth position in PA components in m, shape ``(3,)``.
    r_dot : numpy.ndarray
        Time derivative of ``r`` in the rotating PA frame, m s⁻¹,
        shape ``(3,)``.
    omega : numpy.ndarray
        Mantle angular velocity in PA components, rad s⁻¹, shape ``(3,)``.
    omega_dot : numpy.ndarray
        Mantle angular acceleration in PA components, rad s⁻², shape ``(3,)``.
    parameters : LunarDeformationParameters
        Response parameters.
    radius_m : float
        Lunar reference radius in m.
    earth_mu, moon_mu : float
        Earth and Moon GM in m³ s⁻².

    Returns
    -------
    tidal : numpy.ndarray
        Tidal deformation, shape ``(3, 3)``, normalised by ``M R²``.
    rotational : numpy.ndarray
        Rotational deformation, shape ``(3, 3)``, normalised by ``M R²``.
    derivative : numpy.ndarray
        Time derivative of ``tidal + rotational`` in s⁻¹, shape ``(3, 3)``.
        Disabled parts are zero.

    Notes
    -----
    All inputs refer to the same delayed epoch ``t - τ``.

    References
    ----------
    Folkner et al. (2014), Eqs. 40–42; Park et al. (2021), Eqs. 47–55.
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
    """Delayed deformation along the integrated lunar orbit and attitude.

    One evaluation per step supplies both the mantle inertia used by the
    rotational dynamics and the degree-2 coefficients of the lunar gravity
    field. The degree 2 of ``field`` is replaced, not augmented, by the
    delayed inertia, and the field object itself is never modified.

    Parameters
    ----------
    parameters : LunarDeformationParameters
        Response parameters.
    total_inertia : numpy.ndarray
        Undeformed whole-Moon inertia in PA axes, shape ``(3, 3)``,
        normalised by ``M R²``.
    radius_m : float
        Lunar reference radius in m; equal to the radius of ``field``.
    earth_mu, moon_mu : float
        Earth and Moon GM in m³ s⁻².
    field : GravityField
        Moon gravity field shared with
        :class:`~lunarops.classes.dynamics.force_models.MoonFigureForce` and
        :class:`~lunarops.classes.dynamics.torques.LunarFigureTorque`.
    """

    def __init__(self, parameters, total_inertia, radius_m, earth_mu, moon_mu, field):
        self.parameters = parameters
        self.undistorted_inertia = total_inertia
        self.radius_m = radius_m
        self.earth_mu, self.moon_mu = earth_mu, moon_mu
        self.field = field

    def evaluate(self, epoch_tdb, history, *, rigid=False):
        """Evaluate the delayed inertia and the matching degree-2 gravity.

        Parameters
        ----------
        epoch_tdb : Epoch
            Current TDB epoch ``t``; the response is sampled at ``t - τ``.
        history : DelayedHistory
            Accepted integration history providing the state and its
            derivative at the delayed epoch.
        rigid : bool, default False
            Return the undeformed tensor with zero deformation, as used by
            the rigid backward seed arc.

        Returns
        -------
        LunarInertiaEvaluation
            Delayed inertia, its derivative and the degree-2 coefficients.

        Raises
        ------
        ValueError
            If ``t - τ`` lies outside the accepted history.
        """
        if rigid:
            total = self.undistorted_inertia
            tidal, rotational, derivative = (np.zeros((3, 3)) for _ in range(3))
        else:
            state, state_dot = history.integration_sample(epoch_tdb.shifted(-self.parameters.delay_s))
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
        degree2 = _degree2_coefficients_from_inertia(
            total, c21_static=p.c21_static, s21_static=p.s21_static, s22_static=p.s22_static,
        )
        return LunarInertiaEvaluation(total, derivative, tidal, rotational, degree2)


@dataclass(frozen=True, slots=True)
class LunarDegree2GravityParameters:
    """Parameters of the delayed degree-2 lunar gravity response.

    The defaults are the DE440 values published in the comment area of the
    DE440 SPK file; ``j2_undistorted``, ``beta`` and ``gamma`` describe the
    undeformed body, not the mean observed quadrupole.

    Attributes
    ----------
    j2_undistorted : float
        Unnormalised J2 of the undeformed Moon.
    beta : float
        Moment ratio β = (C - A) / B of the undeformed Moon.
    gamma : float
        Moment ratio γ = (B - A) / C of the undeformed Moon.
    love_k2 : float
        Lunar potential Love number k2.
    tidal_delay_days : float
        Response delay τ in TDB days.
    radius_m : float
        Lunar reference radius in m.
    mean_motion_rad_s : float
        Lunar mean motion in rad s⁻¹.
    include_tidal_deformation : bool
        Include the Earth-raised tidal deformation.
    include_rotational_deformation : bool
        Include the spin deformation.
    c21_static, s21_static, s22_static : float
        Unnormalised static gravity offsets (Pavlov et al. 2016, Eq. 16).

    Raises
    ------
    ValueError
        Unless ``0 < gamma < beta < 1`` and ``j2_undistorted > 0``.
    """

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
        """Validate the moment ratios and J2."""
        if not (0 < self.gamma < self.beta < 1) or self.j2_undistorted <= 0:
            raise ValueError("Lunar deformation requires 0 < gamma < beta < 1 and positive J2.")

    def undistorted_inertia(self) -> np.ndarray:
        """Return the undeformed whole-Moon inertia tensor.

        Returns
        -------
        numpy.ndarray
            Diagonal tensor ``diag(A, B, C)`` in PA axes, normalised by
            ``M R²``, shape ``(3, 3)``, consistent with ``j2_undistorted``,
            ``beta`` and ``gamma``.
        """
        b, g = self.beta, self.gamma
        factor = 2 * self.j2_undistorted / (2 * b - g + b * g)
        return np.diag([factor * (1 - b * g), factor * (1 + g), factor * (1 + b)])


class LunarDegree2GravityModel:
    """Delayed degree-2 lunar gravity along a prescribed orbit and attitude.

    Used when the lunar attitude is not integrated: the deformation is driven
    by the ephemeris Earth-Moon vector and lunar orientation. The orbital
    and spin rates of change are neglected (``r_dot = omega_dot = 0``).

    Parameters
    ----------
    earth_gravitational_parameter_m3_s2 : float
        Earth GM in m³ s⁻².
    moon_gravitational_parameter_m3_s2 : float
        Moon GM in m³ s⁻².
    earth_minus_moon_position_provider : Callable[[Epoch], numpy.ndarray]
        Function of a TDB epoch returning the Earth-minus-Moon position in m
        with ICRF axes, shape ``(3,)``.
    attitude_provider : Callable[[Epoch], numpy.ndarray]
        Function of a TDB epoch returning the PA-to-inertial rotation
        matrix, shape ``(3, 3)``.
    angular_velocity_provider : Callable[[Epoch], numpy.ndarray]
        Function of a TDB epoch returning the lunar angular velocity in PA
        components, rad s⁻¹, shape ``(3,)``.
    parameters : LunarDegree2GravityParameters or None, optional
        Response parameters; the DE440 defaults when ``None``.
    """

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
        """Return the delayed whole-Moon inertia tensor.

        Parameters
        ----------
        epoch : Epoch
            Current epoch; must be TDB. The response is sampled at
            ``epoch - τ``.
        earth_minus_moon_position_provider : Callable, optional
            Replacement for the constructor's position provider, for example
            the integrator's delayed history.

        Returns
        -------
        numpy.ndarray
            Undeformed plus deformed inertia in PA axes, normalised by
            ``M R²``, shape ``(3, 3)``.

        Raises
        ------
        ValueError
            If ``epoch`` is not in TDB.
        """
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
        """Return the fully normalised degree-2 coefficients of the delayed Moon.

        Higher degrees remain those of the static field.

        Parameters
        ----------
        epoch : Epoch
            Current epoch; must be TDB.
        earth_minus_moon_position_provider : Callable, optional
            Replacement for the constructor's position provider.

        Returns
        -------
        numpy.ndarray
            Coefficients of shape ``(2, 3)``: rows C̄ and S̄, columns orders 0–2.

        Raises
        ------
        ValueError
            If ``epoch`` is not in TDB.
        """
        p = self.parameters
        return _degree2_coefficients_from_inertia(
            self.inertia(epoch, earth_minus_moon_position_provider),
            c21_static=p.c21_static, s21_static=p.s21_static, s22_static=p.s22_static,
        )
