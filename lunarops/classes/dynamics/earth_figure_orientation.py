"""Earth gravity-axis models and scalar/batch orientation services."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import erfa
import numpy as np
from lunarops.classes.time import Epoch, TimeScaleConverter, require_tdb_epoch

def inertial2fixed_matrix_from_pole(pole_vector) -> np.ndarray:
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
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class OrientationProvider:
    matrix: Callable[[Epoch], np.ndarray]
    matrices: Callable[[Sequence[Epoch]], np.ndarray]

    @classmethod
    def from_scalar(cls, matrix):
        return cls(matrix, lambda epochs: np.asarray([matrix(epoch) for epoch in epochs]))


_J2000_JD = 2_451_545.0


_DAYS_PER_JULIAN_CENTURY = 36_525.0


DE440_EARTH_POLE_OFFSET_X_ARCSEC = 5.3439916044148049e-3


DE440_EARTH_POLE_OFFSET_Y_ARCSEC = -1.7128824840317532e-2


DE430_PHI_X0_ARCSEC = 5.6754203322893470e-3


DE430_DPHI_X_ARCSEC_PER_YEAR = 2.7689915574483550e-4


DE430_PHI_Y0_ARCSEC = -1.7022656914989530e-2


DE430_DPHI_Y_ARCSEC_PER_YEAR = -1.2118591216559240e-3


def _nutation_matrix(epoch_tdb, node_rate_arcsec, longitude_arcsec, obliquity_arcsec):
    """JPL single-term nutation; coefficients differ between DE430 and DE440."""
    require_tdb_epoch(epoch_tdb)
    days = (epoch_tdb.jd1 - _J2000_JD) + epoch_tdb.jd2
    centuries = days / _DAYS_PER_JULIAN_CENTURY
    node = (450160.280 - node_rate_arcsec * centuries
            + 7.455 * centuries**2 + 0.008 * centuries**3) * erfa.DAS2R
    dpsi = longitude_arcsec * erfa.DAS2R * np.sin(node)
    deps = obliquity_arcsec * erfa.DAS2R * np.cos(node)
    return erfa.numat(erfa.obl80(epoch_tdb.jd1, epoch_tdb.jd2), dpsi, deps)


def de440_earth_pole_vector_inertial(epoch_tdb) -> np.ndarray:
    """Park Eqs. (20)–(26), with ERFA Vondrak precession and JPL offsets."""
    nutation = _nutation_matrix(epoch_tdb, 6962890.539, -17.1996, 9.2025)
    offset_x = DE440_EARTH_POLE_OFFSET_X_ARCSEC * erfa.DAS2R
    offset_y = DE440_EARTH_POLE_OFFSET_Y_ARCSEC * erfa.DAS2R
    corrections = erfa.rx(-offset_x, erfa.ry(-offset_y, np.eye(3)))
    precession = erfa.ltp(erfa.epj(epoch_tdb.jd1, epoch_tdb.jd2))
    # ltp maps J2000 -> date; numat's last row is the true pole in mean-of-date axes.
    pole = precession.T @ corrections @ nutation[2]
    pole /= np.linalg.norm(pole)
    pole.setflags(write=False)
    return pole


def de440_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """DE440 gravity-axis frame; longitude is arbitrary for the zonal field."""
    matrix = np.ascontiguousarray(inertial2fixed_matrix_from_pole(
        de440_earth_pole_vector_inertial(epoch_tdb)).T)
    matrix.setflags(write=False)
    return matrix


def de430_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """Folkner Eqs. (21)–(26): modified nutation and IAU 1976 precession."""
    nutation = _nutation_matrix(epoch_tdb, 6962890.549, -17.206262, 9.205348)
    years = ((epoch_tdb.jd1 - _J2000_JD) + epoch_tdb.jd2) / 365.25
    phi_x = (DE430_PHI_X0_ARCSEC + DE430_DPHI_X_ARCSEC_PER_YEAR * years) * erfa.DAS2R
    phi_y = (DE430_PHI_Y0_ARCSEC + DE430_DPHI_Y_ARCSEC_PER_YEAR * years) * erfa.DAS2R
    corrections = erfa.ry(phi_y, erfa.rx(phi_x, np.eye(3)))
    matrix = np.ascontiguousarray((
        nutation @ corrections @ erfa.pmat76(epoch_tdb.jd1, epoch_tdb.jd2)
    ).T)
    matrix.setflags(write=False)
    return matrix


_TIME = TimeScaleConverter()


def iau2006_earth_pole_vector_inertial(epoch_tdb) -> np.ndarray:
    """Return the model CIP in GCRS axes at the TT corresponding to TDB."""
    tt = _TIME.tdb2tt(epoch_tdb)
    # pnm06a maps GCRS -> true equator/equinox of date; its row 2 is the CIP.
    pole = np.array(erfa.pnm06a(tt.jd1, tt.jd2)[2], copy=True)
    pole /= np.linalg.norm(pole)
    pole.setflags(write=False)
    return pole


def iau2006_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """Gravity axes -> GCRS axes; use the same arbitrary longitude as DE440."""
    matrix = np.ascontiguousarray(inertial2fixed_matrix_from_pole(
        iau2006_earth_pole_vector_inertial(epoch_tdb)).T)
    matrix.setflags(write=False)
    return matrix


EARTH_FIGURE_ORIENTATIONS = {
    "de430": de430_earth_fixed2inertial_matrix,
    "de440": de440_earth_fixed2inertial_matrix,
    "iau2006_2000a": iau2006_earth_fixed2inertial_matrix,
}


def earth_j2_increment(epoch, reference_jd_tdb, linear_per_year, quadratic_per_year2):
    years = ((epoch.jd1 - reference_jd_tdb) + epoch.jd2) / 365.25
    return linear_per_year * years + quadratic_per_year2 * years * years
