"""Spherical-harmonic gravity backed by ``pyshtools.SHGravCoeffs``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pyshtools
except ImportError:  # pragma: no cover
    pyshtools = None


def load_gravity_coefficients(
    file_path, *, file_format="icgem", max_degree=None, gm_m3_s2=None, reference_radius_m=None, name=None
):
    """Load one ``SHGravCoeffs`` object from a supported coefficient file."""
    if pyshtools is None:
        raise ImportError("Loading gravity coefficients requires pyshtools")
    options = {"format": file_format, "lmax": max_degree, "name": name}
    if gm_m3_s2 is not None:
        options["gm"] = float(gm_m3_s2)
    if reference_radius_m is not None:
        options["r0"] = float(reference_radius_m)
    return pyshtools.SHGravCoeffs.from_file(Path(file_path), **options)


class GravityField:
    """Cartesian figure-acceleration adapter around ``SHGravCoeffs``.

    The public ``SHGravCoeffs`` retains its standard degree-0 coefficient.  A
    private coefficient copy has C00 cleared so degree >= 2 can be evaluated
    directly, without subtracting two much larger accelerations.
    """

    def __init__(self, coefficients):
        if pyshtools is None or not isinstance(coefficients, pyshtools.SHGravCoeffs):
            raise TypeError("coefficients must be a pyshtools.SHGravCoeffs object")
        if coefficients.normalization != "4pi" or coefficients.csphase != 1:
            raise ValueError("Gravity coefficients must use 4pi normalization and csphase=1")
        if not np.isfinite(coefficients.gm) or coefficients.gm <= 0:
            raise ValueError("Gravity coefficients require positive finite gm")
        if not np.isfinite(coefficients.r0) or coefficients.r0 <= 0:
            raise ValueError("Gravity coefficients require positive finite r0")
        if not np.isclose(coefficients.coeffs[0, 0, 0], 1.0, rtol=0, atol=1e-15):
            raise ValueError("SHGravCoeffs must use the standard C00=1 convention")
        if coefficients.lmax >= 1 and np.max(np.abs(coefficients.coeffs[:, 1, :2])) > 1e-15:
            raise ValueError("Figure fields must not contain degree-1 terms")
        self.coefficients = coefficients
        self._figure_coefficients = coefficients.copy()
        self._figure_coefficients.set_coeffs(0.0, 0, 0)

    @property
    def gm_m3_s2(self):
        return float(self.coefficients.gm)

    @property
    def radius_m(self):
        return float(self.coefficients.r0)

    def figure_acceleration(self, relative_position_m):
        x = np.asarray(relative_position_m, dtype=float)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            raise ValueError("relative_position_m must be a finite three-vector")
        radius = np.linalg.norm(x)
        if radius == 0:
            raise ValueError("relative_position_m must be nonzero")
        longitude = np.arctan2(x[1], x[0])
        latitude = np.arcsin(np.clip(x[2] / radius, -1.0, 1.0))
        if abs(abs(latitude) - np.pi / 2) <= 1e-12:
            # Avoid entering the Fortran routine at its exact pole singularity.
            # The angular offset must exceed sqrt(machine epsilon), otherwise
            # x/r still rounds to an exact pole before arcsin.
            epsilon = max(radius * 1e-6, 1e-2)
            return 0.5 * (
                self.figure_acceleration(x + np.array((epsilon, 0.0, 0.0)))
                + self.figure_acceleration(x - np.array((epsilon, 0.0, 0.0)))
            )
        try:
            spherical = self._figure_coefficients.expand(
                lat=np.array([latitude]),
                lon=np.array([longitude]),
                r=np.array([radius]),
                degrees=False,
                normal_gravity=False,
                omega=0.0,
            )[0]
        except ValueError:
            # PlmBar_d1 is singular at the exact poles. Evaluate the limiting
            # Cartesian field from two nearby longitudes instead.
            if abs(abs(latitude) - np.pi / 2) > 1e-12:
                raise
            epsilon = max(radius * 1e-6, 1e-2)
            return 0.5 * (
                self.figure_acceleration(x + np.array((epsilon, 0.0, 0.0)))
                + self.figure_acceleration(x - np.array((epsilon, 0.0, 0.0)))
            )
        e_r = np.array((np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)))
        # theta is colatitude, so +theta points southward.
        e_theta = np.array(
            (np.sin(latitude) * np.cos(longitude), np.sin(latitude) * np.sin(longitude), -np.cos(latitude))
        )
        e_phi = np.array((-np.sin(longitude), np.cos(longitude), 0.0))
        return spherical[0] * e_r + spherical[1] * e_theta + spherical[2] * e_phi

    def figure_accelerations(self, relative_positions_m):
        """Evaluate figure accelerations for several source-centred points.

        Parameters
        ----------
        relative_positions_m : array-like, shape (N, 3)
            Positions relative to the centre of this gravity field, expressed
            in the field's body-fixed frame.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Cartesian accelerations in the same body-fixed frame.  The
            spherical-harmonic expansion is submitted to pyshtools once for
            the complete target batch instead of once per target.

        Notes
        -----
        ``SHGravCoeffs.expand`` has a singular associated-Legendre evaluation
        exactly at the poles.  Pole points are uncommon, so they are handled
        through the scalar limiting evaluation while all regular points stay
        on the batched fast path.
        """
        x = np.asarray(relative_positions_m, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
            raise ValueError("relative_positions_m must have shape (N,3) and be finite")
        if len(x) == 0:
            return np.empty((0, 3), dtype=float)
        radius = np.linalg.norm(x, axis=1)
        if np.any(radius == 0):
            raise ValueError("relative_positions_m must be nonzero")

        longitude = np.arctan2(x[:, 1], x[:, 0])
        latitude = np.arcsin(np.clip(x[:, 2] / radius, -1.0, 1.0))
        pole = np.abs(np.abs(latitude) - np.pi / 2) <= 1e-12

        result = np.empty_like(x)
        regular = ~pole
        if np.any(regular):
            lat = latitude[regular]
            lon = longitude[regular]
            spherical = self._figure_coefficients.expand(
                lat=lat,
                lon=lon,
                r=radius[regular],
                degrees=False,
                normal_gravity=False,
                omega=0.0,
            )
            cos_lat, sin_lat = np.cos(lat), np.sin(lat)
            cos_lon, sin_lon = np.cos(lon), np.sin(lon)
            e_r = np.column_stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat))
            # theta is colatitude, so +theta points southward.
            e_theta = np.column_stack((sin_lat * cos_lon, sin_lat * sin_lon, -cos_lat))
            e_phi = np.column_stack((-sin_lon, cos_lon, np.zeros_like(lon)))
            result[regular] = (
                spherical[:, 0, None] * e_r + spherical[:, 1, None] * e_theta + spherical[:, 2, None] * e_phi
            )
        if np.any(pole):
            # Reuse the tested scalar limiting path for the rare exact-pole
            # points; this also covers backend-specific pole exceptions.
            result[pole] = np.vstack([self.figure_acceleration(point) for point in x[pole]])
        return result


def make_gravity_coefficients(cosine, sine, *, gm_m3_s2, radius_m, name):
    """Construct a standard ``SHGravCoeffs`` field with ``C00=1``."""
    if pyshtools is None:
        raise ImportError("Spherical-harmonic gravity requires pyshtools")
    array = np.stack((np.asarray(cosine, float), np.asarray(sine, float)))
    field = pyshtools.SHGravCoeffs.from_array(
        array,
        gm=float(gm_m3_s2),
        r0=float(radius_m),
        normalization="4pi",
        csphase=1,
        name=name,
    )
    return field


def make_j2_coefficients(*, gm_m3_s2, radius_m, j2, name=None):
    """Return a degree-2 gravity field equivalent to an axisymmetric J2."""
    if not np.isfinite(j2):
        raise ValueError("j2 must be finite")
    cosine = np.zeros((3, 3))
    sine = np.zeros_like(cosine)
    cosine[2, 0] = -float(j2) / np.sqrt(5.0)
    return make_gravity_coefficients(
        cosine,
        sine,
        gm_m3_s2=gm_m3_s2,
        radius_m=radius_m,
        name="axisymmetric J2" if name is None else name,
    )


def frame_from_pole(pole_inertial):
    """Return a passive inertial-to-body rotation with body z along ``pole``."""
    pole = np.asarray(pole_inertial, dtype=float)
    if (
        pole.shape != (3,)
        or not np.all(np.isfinite(pole))
        or not np.isclose(np.linalg.norm(pole), 1.0, rtol=0, atol=1e-14)
    ):
        raise ValueError("pole_inertial must be a finite unit vector")
    reference = np.array((0.0, 0.0, 1.0))
    if abs(np.dot(reference, pole)) > 0.9:
        reference = np.array((1.0, 0.0, 0.0))
    x_axis = np.cross(reference, pole)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(pole, x_axis)
    frame = np.vstack((x_axis, y_axis, pole))
    frame.setflags(write=False)
    return frame


def as_gravity_field(coefficients):
    return coefficients if isinstance(coefficients, GravityField) else GravityField(coefficients)
