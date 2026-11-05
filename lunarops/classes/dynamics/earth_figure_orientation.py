"""Orientation of the Earth's gravity-field axes.

The Earth figure force and torque use only the zonal Earth field, so the
gravity-field frame needs only the true pole: z is along the pole and the
longitude of x is arbitrary. Three pole models are available, keyed in
:data:`EARTH_FIGURE_ORIENTATIONS`:

- ``"de430"``: IAU 1976 precession, a one-term nutation and linearly drifting
  pole offsets (Folkner et al. 2014, Eqs. 21–26);
- ``"de440"``: Vondrák et al. (2011) long-term precession, a one-term
  nutation and constant pole offsets (Park et al. 2021, Eqs. 20–26);
- ``"iau2006_2000a"``: the IAU 2006/2000A celestial intermediate pole.

All functions take TDB epochs and return rotations from gravity-field axes
to ICRF (GCRS) axes.

Attributes
----------
DE440_EARTH_POLE_OFFSET_X_ARCSEC, DE440_EARTH_POLE_OFFSET_Y_ARCSEC : float
    Constant DE440 pole offsets in arcsec.
DE430_PHI_X0_ARCSEC, DE430_PHI_Y0_ARCSEC : float
    DE430 pole offsets at J2000 in arcsec.
DE430_DPHI_X_ARCSEC_PER_YEAR, DE430_DPHI_Y_ARCSEC_PER_YEAR : float
    DE430 pole-offset rates in arcsec per Julian year.
EARTH_FIGURE_ORIENTATIONS : dict
    Gravity-axes-to-inertial matrix function per pole-model name.
"""
from __future__ import annotations

import erfa
import numpy as np

from lunarops.classes.time import TimeScaleConverter, require_tdb_epoch


def inertial2fixed_matrix_from_pole(pole_vector) -> np.ndarray:
    """Return a rotation from inertial axes to axes with z along a pole.

    The x axis is the inertial x axis (or y axis when the pole lies within
    about 25° of x) projected onto the plane normal to the pole; y completes
    the right-handed triad.

    Parameters
    ----------
    pole_vector : array_like
        Pole direction in inertial axes, shape ``(3,)``; any nonzero norm.

    Returns
    -------
    numpy.ndarray
        Matrix with rows ``(x, y, z)`` of the body axes, so that
        ``v_fixed = M @ v_inertial``, shape ``(3, 3)``.

    Raises
    ------
    ValueError
        If ``pole_vector`` is not a finite nonzero three-vector.
    """
    pole = np.asarray(pole_vector, dtype=float)
    if pole.shape != (3,) or not np.all(np.isfinite(pole)):
        raise ValueError("pole_vector must be a finite three-vector")
    norm = np.linalg.norm(pole)
    if norm == 0.0:
        raise ValueError("pole_vector must be nonzero")
    z_axis = pole / norm
    reference = np.array((1.0, 0.0, 0.0))
    if abs(float(reference @ z_axis)) > 0.9:
        reference = np.array((0.0, 1.0, 0.0))
    x_axis = reference - (reference @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.ascontiguousarray(np.vstack((x_axis, y_axis, z_axis)))
    return matrix


_J2000_JD = 2_451_545.0


_DAYS_PER_JULIAN_CENTURY = 36_525.0


DE440_EARTH_POLE_OFFSET_X_ARCSEC = 5.3439916044148049e-3


DE440_EARTH_POLE_OFFSET_Y_ARCSEC = -1.7128824840317532e-2


DE430_PHI_X0_ARCSEC = 5.6754203322893470e-3


DE430_DPHI_X_ARCSEC_PER_YEAR = 2.7689915574483550e-4


DE430_PHI_Y0_ARCSEC = -1.7022656914989530e-2


DE430_DPHI_Y_ARCSEC_PER_YEAR = -1.2118591216559240e-3


def _nutation_matrix(epoch_tdb, node_rate_arcsec, longitude_arcsec, obliquity_arcsec):
    """Return the one-term (18.6-year) nutation matrix of a TDB epoch."""
    require_tdb_epoch(epoch_tdb)
    days = (epoch_tdb.jd1 - _J2000_JD) + epoch_tdb.jd2
    centuries = days / _DAYS_PER_JULIAN_CENTURY
    node = (450160.280 - node_rate_arcsec * centuries
            + 7.455 * centuries**2 + 0.008 * centuries**3) * erfa.DAS2R
    dpsi = longitude_arcsec * erfa.DAS2R * np.sin(node)
    deps = obliquity_arcsec * erfa.DAS2R * np.cos(node)
    return erfa.numat(erfa.obl80(epoch_tdb.jd1, epoch_tdb.jd2), dpsi, deps)


def de440_earth_pole_vector_inertial(epoch_tdb) -> np.ndarray:
    """Return the DE440 Earth true pole in ICRF axes.

    Parameters
    ----------
    epoch_tdb : Epoch
        Evaluation epoch; must be TDB.

    Returns
    -------
    numpy.ndarray
        Unit pole vector, shape ``(3,)``.

    Raises
    ------
    ValueError
        If ``epoch_tdb`` is not in TDB.

    References
    ----------
    Park et al. (2021), Eqs. 20–26; Vondrák et al. (2011) precession.
    """
    nutation = _nutation_matrix(epoch_tdb, 6962890.539, -17.1996, 9.2025)
    offset_x = DE440_EARTH_POLE_OFFSET_X_ARCSEC * erfa.DAS2R
    offset_y = DE440_EARTH_POLE_OFFSET_Y_ARCSEC * erfa.DAS2R
    corrections = erfa.rx(-offset_x, erfa.ry(-offset_y, np.eye(3)))
    precession = erfa.ltp(erfa.epj(epoch_tdb.jd1, epoch_tdb.jd2))
    # ltp maps J2000 -> date; numat's last row is the true pole in mean-of-date axes.
    pole = precession.T @ corrections @ nutation[2]
    pole /= np.linalg.norm(pole)
    return pole


def de440_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """Return the DE440 Earth gravity-axes-to-ICRF rotation.

    Parameters
    ----------
    epoch_tdb : Epoch
        Evaluation epoch; must be TDB.

    Returns
    -------
    numpy.ndarray
        Matrix with ``v_inertial = M @ v_fixed``, shape ``(3, 3)``. Its third
        column is the DE440 pole; the x-axis longitude is arbitrary.

    Raises
    ------
    ValueError
        If ``epoch_tdb`` is not in TDB.
    """
    matrix = np.ascontiguousarray(inertial2fixed_matrix_from_pole(
        de440_earth_pole_vector_inertial(epoch_tdb)).T)
    return matrix


def de430_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """Return the DE430 Earth gravity-axes-to-ICRF rotation.

    Parameters
    ----------
    epoch_tdb : Epoch
        Evaluation epoch; must be TDB.

    Returns
    -------
    numpy.ndarray
        Matrix with ``v_inertial = M @ v_fixed``, shape ``(3, 3)``.

    Raises
    ------
    ValueError
        If ``epoch_tdb`` is not in TDB.

    References
    ----------
    Folkner et al. (2014), Eqs. 21–26.
    """
    nutation = _nutation_matrix(epoch_tdb, 6962890.549, -17.206262, 9.205348)
    years = ((epoch_tdb.jd1 - _J2000_JD) + epoch_tdb.jd2) / 365.25
    phi_x = (DE430_PHI_X0_ARCSEC + DE430_DPHI_X_ARCSEC_PER_YEAR * years) * erfa.DAS2R
    phi_y = (DE430_PHI_Y0_ARCSEC + DE430_DPHI_Y_ARCSEC_PER_YEAR * years) * erfa.DAS2R
    corrections = erfa.ry(phi_y, erfa.rx(phi_x, np.eye(3)))
    matrix = np.ascontiguousarray((
        nutation @ corrections @ erfa.pmat76(epoch_tdb.jd1, epoch_tdb.jd2)
    ).T)
    return matrix


_TIME = TimeScaleConverter()


def iau2006_earth_pole_vector_inertial(epoch_tdb) -> np.ndarray:
    """Return the IAU 2006/2000A celestial intermediate pole in GCRS axes.

    Parameters
    ----------
    epoch_tdb : Epoch
        Evaluation epoch in TDB; it is converted to TT for the model.

    Returns
    -------
    numpy.ndarray
        Unit CIP vector, shape ``(3,)``.
    """
    tt = _TIME.tdb2tt(epoch_tdb)
    # pnm06a maps GCRS -> true equator/equinox of date; its row 2 is the CIP.
    pole = erfa.pnm06a(tt.jd1, tt.jd2)[2]
    pole /= np.linalg.norm(pole)
    return pole


def iau2006_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """Return the IAU 2006/2000A Earth gravity-axes-to-GCRS rotation.

    The x-axis longitude follows the same convention as
    :func:`de440_earth_fixed2inertial_matrix`.

    Parameters
    ----------
    epoch_tdb : Epoch
        Evaluation epoch in TDB.

    Returns
    -------
    numpy.ndarray
        Matrix with ``v_gcrs = M @ v_fixed``, shape ``(3, 3)``.
    """
    matrix = np.ascontiguousarray(inertial2fixed_matrix_from_pole(
        iau2006_earth_pole_vector_inertial(epoch_tdb)).T)
    return matrix


EARTH_FIGURE_ORIENTATIONS = {
    "de430": de430_earth_fixed2inertial_matrix,
    "de440": de440_earth_fixed2inertial_matrix,
    "iau2006_2000a": iau2006_earth_fixed2inertial_matrix,
}


def earth_j2_increment(epoch, reference_jd_tdb, linear_per_year, quadratic_per_year2):
    """Return the time-variable increment of the Earth J2.

    Parameters
    ----------
    epoch : Epoch
        Evaluation epoch; its two-part Julian date is read as TDB.
    reference_jd_tdb : float
        Reference epoch of the polynomial as a TDB Julian date.
    linear_per_year : float
        Linear coefficient per Julian year.
    quadratic_per_year2 : float
        Quadratic coefficient per Julian year squared (no factor 1/2).

    Returns
    -------
    float
        ``linear · t + quadratic · t²`` with ``t`` in Julian years since the
        reference epoch (dimensionless J2 increment).
    """
    years = ((epoch.jd1 - reference_jd_tdb) + epoch.jd2) / 365.25
    return linear_per_year * years + quadratic_per_year2 * years * years
