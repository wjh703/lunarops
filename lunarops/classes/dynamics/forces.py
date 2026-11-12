"""Parameters of the solar source terms and the Earth-tide model.

:class:`SolarSourceParameters` holds the solar constants of the
Lense-Thirring and radiation-pressure terms. :class:`EarthTideModel` prepares
the delayed tide-raiser vectors of the Earth-tide acceleration; the
acceleration itself, like every other force term, is evaluated by the native
kernel in :mod:`lunarops._dynamics_core`.

Notes
-----
The EIH term (Park et al. 2021, Eq. 27, with ``R_ij = r_j - r_i``) uses the
Newtonian point-mass sums over all bodies, prescribed sources included, as
its lower-order accelerations; figure and tide accelerations do not enter
the 1PN term.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType

import erfa
import numpy as np

from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.classes.ephemerides.body_ids import body_name


@dataclass(frozen=True, slots=True)
class SolarSourceParameters:
    """Solar constants of the Lense-Thirring and radiation-pressure terms.

    Attributes
    ----------
    gravitational_parameter_m3_s2 : float
        Solar GM in m³ s⁻².
    radius_m : float, default 696_000_000.0
        Solar radius in m.
    spin_rate_rad_s : float
        Solar rotation rate in rad s⁻¹; 14.1844° per day by default.
    spin_axis : tuple of float, default (0.0, 0.0, 1.0)
        Solar spin axis with ICRF axes; it is not normalised.
    gamma : float, default 1.0
        PPN parameter γ of the Lense-Thirring term.
    moment_factor : float, default 0.06884
        Dimensionless solar moment of inertia ``C / (M R²)``.

    Notes
    -----
    The derived angular momentum per unit gravitational constant,
    ``moment_factor · GM · R² · ω · axis`` in m⁵ s⁻³, is stored privately for
    the native kernel.
    """

    gravitational_parameter_m3_s2: float
    radius_m: float = 696_000_000.0
    spin_rate_rad_s: float = np.deg2rad(14.1844) / SECONDS_PER_DAY
    spin_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gamma: float = 1.0
    moment_factor: float = 0.06884
    _angular_momentum: np.ndarray = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Store the spin axis as floats and derive the angular momentum."""
        axis = np.asarray(self.spin_axis, dtype=float)
        object.__setattr__(self, "spin_axis", tuple(float(x) for x in axis))
        angular_momentum = (
            self.moment_factor * self.gravitational_parameter_m3_s2 * self.radius_m**2 * self.spin_rate_rad_s * axis
        )
        object.__setattr__(self, "_angular_momentum", angular_momentum)


@dataclass(frozen=True, slots=True)
class EarthTideParameters:
    """Earth-tide Love numbers and time delays.

    The defaults are the DE440 values. Index ``j`` of each delay tuple refers
    to tidal order ``j`` (0: zonal, 1: diurnal, 2: semidiurnal).

    Attributes
    ----------
    k20, k21, k22 : float
        Degree-2 potential Love numbers of orders 0, 1 and 2.
    tau_orb_days : tuple of float
        Orbital time delays τ_orb,j in TDB days; the raiser positions are
        taken at ``t - τ_orb,j``.
    tau_rot_days : tuple of float
        Rotational time delays τ_rot,j in days; the raiser vectors are
        rotated about the Earth pole by ``-ω_E τ_rot,j``.
    earth_rotation_rate_rad_s : float
        Earth rotation rate ω_E in rad s⁻¹.

    References
    ----------
    Park et al. (2021), Eqs. 35–36.
    """

    k20: float = 0.335
    k21: float = 0.277
    k22: float = 0.283
    tau_orb_days: tuple[float, float, float] = (0.077, -0.010, -0.20982243826832672)
    tau_rot_days: tuple[float, float, float] = (0.0, 0.011929122823877609, -0.0007423627529883070)
    earth_rotation_rate_rad_s: float = 7.29211514670698e-5

    def __post_init__(self):
        """Convert the delay tuples to floats."""
        tau_orb_days = tuple(float(value) for value in self.tau_orb_days)
        tau_rot_days = tuple(float(value) for value in self.tau_rot_days)
        object.__setattr__(self, "tau_orb_days", tau_orb_days)
        object.__setattr__(self, "tau_rot_days", tau_rot_days)


class EarthTideModel:
    """Delayed Earth tide raised by the Moon and the Sun.

    The model is evaluated as the relative Moon-minus-Earth acceleration.
    :class:`~lunarops.classes.dynamics.force_models.EarthTideForce` applies
    the Moon's barycentric share ``GM_E / (GM_E + GM_M)`` of it to the Moon
    and the remainder, with opposite sign, to the Earth.

    Parameters
    ----------
    earth_gravitational_parameter_m3_s2 : float
        Earth GM in m³ s⁻².
    moon_gravitational_parameter_m3_s2 : float
        Moon GM in m³ s⁻².
    tide_raiser_gm : Mapping[str, float]
        GM in m³ s⁻² of every tide raiser, keyed by body identifier.
    earth_radius_m : float, default 6_378_136.6
        Earth reference radius in m.
    tide_parameters : EarthTideParameters or None, optional
        Love numbers and delays; the DE440 defaults when ``None``.
    tide_raisers : sequence of str, default ("MOON", "SUN")
        Bodies raising the tide.
    solar_gm_variation : SolarGmVariation or None, optional
        Solar GM drift applied to the Sun as a raiser.

    Raises
    ------
    KeyError
        If a raiser is missing from ``tide_raiser_gm``.

    References
    ----------
    Park et al. (2021), Eqs. 35–36.
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
        self.tide_parameters = EarthTideParameters() if tide_parameters is None else tide_parameters
        self.tide_raisers = tuple(body_name(name) for name in tide_raisers)
        raiser_gm = {body_name(name): float(value) for name, value in tide_raiser_gm.items()}
        self.tide_raiser_gm = MappingProxyType(raiser_gm)
        self.solar_gm_variation = solar_gm_variation
        tide = self.tide_parameters
        self._delay_seconds = tuple(-tide.tau_orb_days[order] * SECONDS_PER_DAY for order in range(3))
        self._rotation_lags = tuple(
            erfa.rz(-tide.earth_rotation_rate_rad_s * tide.tau_rot_days[order] * SECONDS_PER_DAY, np.eye(3))
            for order in range(3)
        )
        self._base_raiser_gm = np.ascontiguousarray([raiser_gm[name] for name in self.tide_raisers], dtype=float)
        self._sun_row = self.tide_raisers.index("SUN") if "SUN" in self.tide_raisers else None

    def delayed_vectors(self, epoch, inertial2fixed_matrix, history):
        """Return the raiser vectors at the three delayed epochs.

        For each tidal order ``j`` the geocentric raiser position at
        ``t - τ_orb,j`` is expressed in the Earth gravity-field axes of the
        current epoch and rotated by the rotational lag ``-ω_E τ_rot,j``.

        Parameters
        ----------
        epoch : Epoch
            Current TDB epoch.
        inertial2fixed_matrix : numpy.ndarray
            Rotation from ICRF to Earth gravity-field axes at ``epoch``,
            shape ``(3, 3)``.
        history : StateHistoryProvider
            Source of delayed BCRS states.

        Returns
        -------
        delayed_vectors : numpy.ndarray
            Earth-to-raiser vectors in m, shape ``(n_raisers, 3, 3)``
            indexed by raiser, tidal order and axis.
        raiser_gm : numpy.ndarray
            Raiser GM in m³ s⁻² at ``epoch``, including the solar GM drift,
            shape ``(n_raisers,)``.
        """
        delayed_vectors = np.empty((len(self.tide_raisers), 3, 3), dtype=float)
        # One history query for the three delayed epochs; the Moon's vector is the relative state itself.
        geocentric = history.geocentric_positions(
            self.tide_raisers, [epoch.shifted(delay) for delay in self._delay_seconds]
        )
        for order in range(3):
            # Eq. (36) combines these vectors with the current Earth-Moon
            # vector, so every operand must use the current fixed-frame basis.
            lagged_fixed = self._rotation_lags[order] @ inertial2fixed_matrix
            delayed_vectors[:, order] = geocentric[order] @ lagged_fixed.T
        raiser_gm = self._base_raiser_gm
        if self.solar_gm_variation is not None and self._sun_row is not None:
            raiser_gm = raiser_gm.copy()
            raiser_gm[self._sun_row] *= self.solar_gm_variation.factor(epoch)
        return delayed_vectors, raiser_gm
