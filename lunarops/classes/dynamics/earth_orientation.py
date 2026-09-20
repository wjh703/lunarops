"""DE440 long-term Earth pole used by the translational dynamics."""

from __future__ import annotations

import erfa
import numpy as np

from lunarops.classes.ephemerides import require_tdb_epoch

from .gravity import frame_from_pole

_J2000_JD = 2_451_545.0
_DAYS_PER_JULIAN_CENTURY = 36_525.0
_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)

# DE440 integration constants.  The angles are EME2000-to-ICRF pole offsets
# in arcseconds; their fitted rates are zero in DE440.
DE440_EARTH_POLE_OFFSET_X_ARCSEC = 5.3439916044148049e-3
DE440_EARTH_POLE_OFFSET_Y_ARCSEC = -1.7128824840317532e-2


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def de440_earth_pole_inertial(epoch_tdb) -> np.ndarray:
    """Return the DE440 Earth pole in ICRF axes (Park et al. 2021, 20-26)."""
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
    precession_to_inertial = np.column_stack((x_axis, y_axis, equatorial_pole))

    offset_x = DE440_EARTH_POLE_OFFSET_X_ARCSEC * _ARCSEC_TO_RAD
    offset_y = DE440_EARTH_POLE_OFFSET_Y_ARCSEC * _ARCSEC_TO_RAD
    pole = precession_to_inertial @ _rotation_x(-offset_x) @ _rotation_y(-offset_y) @ true_pole_of_date
    pole /= np.linalg.norm(pole)
    pole.setflags(write=False)
    return pole


def de440_earth_dynamics_frame(epoch_tdb) -> np.ndarray:
    """Return an inertial-to-equatorial frame whose z-axis is the DE440 pole."""
    return frame_from_pole(de440_earth_pole_inertial(epoch_tdb))


__all__ = [
    "DE440_EARTH_POLE_OFFSET_X_ARCSEC",
    "DE440_EARTH_POLE_OFFSET_Y_ARCSEC",
    "de440_earth_dynamics_frame",
    "de440_earth_pole_inertial",
]
