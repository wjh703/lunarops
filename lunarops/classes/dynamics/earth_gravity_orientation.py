"""DE430/DE440 Earth gravity-axis orientation models for lunar dynamics."""

from __future__ import annotations

import erfa
import numpy as np

from lunarops.classes.time import require_tdb_epoch

from .orientation import inertial2fixed_matrix_from_pole

_J2000_JD = 2_451_545.0
_DAYS_PER_JULIAN_CENTURY = 36_525.0
_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)

# DE440 integration constants.  The angles are EME2000-to-ICRF pole offsets
# in arcseconds; their fitted rates are zero in DE440.
DE440_EARTH_POLE_OFFSET_X_ARCSEC = 5.3439916044148049e-3
DE440_EARTH_POLE_OFFSET_Y_ARCSEC = -1.7128824840317532e-2
DE430_PHI_X0_ARCSEC = 5.6754203322893470e-3
DE430_DPHI_X_ARCSEC_PER_YEAR = 2.7689915574483550e-4
DE430_PHI_Y0_ARCSEC = -1.7022656914989530e-2
DE430_DPHI_Y_ARCSEC_PER_YEAR = -1.2118591216559240e-3


def _passive_rotation_x(angle: float) -> np.ndarray:
    """Return the passive rotation about x used by the DE430 equations."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, s), (0.0, -s, c)))


def _passive_rotation_y(angle: float) -> np.ndarray:
    """Return the passive rotation about y used by the DE430 equations."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c)))


def de440_earth_pole_vector_inertial(epoch_tdb) -> np.ndarray:
    """Return the DE440 Earth-pole unit vector in ICRF coordinates."""
    require_tdb_epoch(epoch_tdb)
    jd = float(epoch_tdb.jd1 + epoch_tdb.jd2)
    centuries = (jd - _J2000_JD) / _DAYS_PER_JULIAN_CENTURY

    node_arcsec = (
        125.0 * 3600.0
        + 2.0 * 60.0
        + 40.280
        - (1934.0 * 3600.0 + 8.0 * 60.0 + 10.539) * centuries
        + 7.455 * centuries**2
        + 0.008 * centuries**3
    )
    node = node_arcsec * _ARCSEC_TO_RAD
    delta_psi = -17.1996 * _ARCSEC_TO_RAD * np.sin(node)
    delta_epsilon = 9.2025 * _ARCSEC_TO_RAD * np.cos(node)
    mean_epsilon = (84_381.448 - 46.815 * centuries - 0.00059 * centuries**2 + 0.001813 * centuries**3) * _ARCSEC_TO_RAD

    true_pole_of_date = np.array(
        (
            np.sin(delta_psi) * np.sin(mean_epsilon + delta_epsilon),
            np.cos(delta_psi) * np.sin(mean_epsilon + delta_epsilon) * np.cos(mean_epsilon)
            - np.cos(mean_epsilon + delta_epsilon) * np.sin(mean_epsilon),
            np.cos(delta_psi) * np.sin(mean_epsilon + delta_epsilon) * np.sin(mean_epsilon)
            + np.cos(mean_epsilon + delta_epsilon) * np.cos(mean_epsilon),
        )
    )

    julian_epoch = 2000.0 + (jd - _J2000_JD) / 365.25
    ecliptic_pole = np.asarray(erfa.ltpecl(julian_epoch), dtype=float)
    equatorial_pole = np.asarray(erfa.ltpequ(julian_epoch), dtype=float)
    x_axis = np.cross(equatorial_pole, ecliptic_pole)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(equatorial_pole, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    date_equatorial2icrf_matrix = np.column_stack((x_axis, y_axis, equatorial_pole))

    offset_x = DE440_EARTH_POLE_OFFSET_X_ARCSEC * _ARCSEC_TO_RAD
    offset_y = DE440_EARTH_POLE_OFFSET_Y_ARCSEC * _ARCSEC_TO_RAD
    # DE430 specifies passive rotations with the negative offset angles.
    pole = (
        date_equatorial2icrf_matrix
        @ _passive_rotation_x(-offset_x)
        @ _passive_rotation_y(-offset_y)
        @ true_pole_of_date
    )
    pole /= np.linalg.norm(pole)
    pole.setflags(write=False)
    return pole


def de440_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """Return the DE440 Earth-fixed-to-ICRF rotation matrix."""
    inertial2fixed = inertial2fixed_matrix_from_pole(de440_earth_pole_vector_inertial(epoch_tdb))
    matrix = np.ascontiguousarray(inertial2fixed.T)
    matrix.setflags(write=False)
    return matrix


def _passive_rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))


def de430_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """Return the DE430 Earth-fixed-to-inertial matrix from Folkner Eq. 25."""
    require_tdb_epoch(epoch_tdb)
    days = float(epoch_tdb.jd1 + epoch_tdb.jd2) - _J2000_JD
    centuries = days / _DAYS_PER_JULIAN_CENTURY
    omega = np.deg2rad(
        125.0 + 2.0 / 60.0 + 40.280 / 3600.0
        + (-(1934.0 + 8.0 / 60.0 + 10.539 / 3600.0) * centuries
           + 7.455 / 3600.0 * centuries**2 + 0.008 / 3600.0 * centuries**3)
    )
    delta_psi = -17.1996 * _ARCSEC_TO_RAD * np.sin(omega)
    delta_epsilon = 9.2025 * _ARCSEC_TO_RAD * np.cos(omega)
    mean_epsilon = (84381.448 - 46.815 * centuries - 0.00059 * centuries**2
                    + 0.001813 * centuries**3) * _ARCSEC_TO_RAD
    nutation = (
        _passive_rotation_x(-(mean_epsilon + delta_epsilon))
        @ _passive_rotation_z(-delta_psi)
        @ _passive_rotation_x(mean_epsilon)
    )
    zeta_a = (2306.2181 * centuries + 0.30188 * centuries**2 + 0.017998 * centuries**3) * _ARCSEC_TO_RAD
    theta_a = (2004.3109 * centuries - 0.42665 * centuries**2 - 0.041833 * centuries**3) * _ARCSEC_TO_RAD
    z_a = (2306.2181 * centuries + 1.09468 * centuries**2 + 0.018203 * centuries**3) * _ARCSEC_TO_RAD
    phi_x = (DE430_PHI_X0_ARCSEC + DE430_DPHI_X_ARCSEC_PER_YEAR * (days / 365.25)) * _ARCSEC_TO_RAD
    phi_y = (DE430_PHI_Y0_ARCSEC + DE430_DPHI_Y_ARCSEC_PER_YEAR * (days / 365.25)) * _ARCSEC_TO_RAD
    precession = _passive_rotation_z(-z_a) @ _passive_rotation_y(theta_a) @ _passive_rotation_z(-zeta_a)
    corrections = _passive_rotation_y(phi_y) @ _passive_rotation_x(phi_x)
    inertial2fixed = nutation @ corrections @ precession
    fixed2inertial = np.ascontiguousarray(inertial2fixed.T)
    fixed2inertial.setflags(write=False)
    return fixed2inertial


__all__ = [
    "DE440_EARTH_POLE_OFFSET_X_ARCSEC",
    "DE440_EARTH_POLE_OFFSET_Y_ARCSEC",
    "de430_earth_fixed2inertial_matrix",
    "de440_earth_fixed2inertial_matrix",
    "de440_earth_pole_vector_inertial",
]
