"""IERS 2010 ocean pole tide loading displacement and its coefficient grid."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TextIO

import numpy as np
from numpy.typing import ArrayLike

from lunarops.classes.frames.earth_orientation import EarthOrientationProvider
from lunarops.config.schema import ConfigSchema, path

from .base import StationDisplacementInput
from .pole_tide import polar_wobble
from .terrestrial_geometry import enu2itrf, itrf2geodetic

_GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67428e-11
_EARTH_EQUATORIAL_RADIUS_M = 6_378_136.6
_EARTH_GM_M3_S2 = 3.986004418e14
_EARTH_ANGULAR_VELOCITY_RAD_S = 7.292115e-5
_EQUATORIAL_GRAVITY_M_S2 = 9.7803278
_SEAWATER_DENSITY_KG_M3 = 1025.0
_LOAD_LOVE_COMBINATION = 0.6870 + 0.0036j

_DIGITS = r"\d+(?:_\d+)*"
_NUMBER = (
    rf"[+-]?(?:[0-9]+\.[0-9]+|(?:{_DIGITS}(?:\.(?:{_DIGITS})?)?|\.{_DIGITS})"
    rf"(?:[eE][+-]?{_DIGITS})?|(?ai:inf(?:inity)?|nan))"
)
_COEFFICIENT_ROW = re.compile(
    r"^[^\S\r\n]*" + r"[^\S\r\n]+".join([f"({_NUMBER})"] * 8) + r"(?=[^\S\r\n]|$)", re.MULTILINE
)


@dataclass(frozen=True, slots=True)
class OceanPoleTideCoefficients:
    """Ocean pole tide loading coefficients interpolated at one station.

    Attributes
    ----------
    latitude_rad : float
        Geodetic latitude of the station (WGS84), north positive, in radians.
    longitude_rad : float
        Longitude of the station, east positive, in radians.
    ellipsoidal_height_m : float
        Height of the station above the WGS84 ellipsoid, in metres.
    radial, north, east : complex
        Complex loading coefficients u^R + i u^I of the up, north and east components (dimensionless).
    """

    latitude_rad: float
    longitude_rad: float
    ellipsoidal_height_m: float
    radial: complex
    north: complex
    east: complex


class OceanPoleTideGrid:
    """Regular global grid of ocean pole tide loading coefficients with bilinear interpolation.

    Each row of the coefficient file holds longitude and latitude in degrees followed by the real and
    imaginary parts of the radial (up), north and east coefficients. Plain and gzip-compressed files are
    accepted. Interpolation clamps the latitude to the grid range and wraps the longitude by 360 degrees.

    Parameters
    ----------
    coefficient_file : str or Path
        Coefficient grid file (``opoleloadcoefcmcor.txt`` format), optionally ``.gz``.

    Attributes
    ----------
    coefficient_file : Path
        Resolved coefficient file.
    longitude_grid_deg, latitude_grid_deg : ndarray
        Grid node coordinates, in degrees, ascending.
    radial_grid, north_grid, east_grid : ndarray, shape (n_latitude, n_longitude)
        Complex coefficients of the up, north and east components.
    longitude_step_deg, latitude_step_deg : float
        Grid spacing, in degrees.
    longitude_min_deg, longitude_max_deg, latitude_min_deg, latitude_max_deg : float
        Grid extent, in degrees.

    Raises
    ------
    FileNotFoundError
        If the coefficient file does not exist.
    ValueError
        If the file holds no coefficient rows or does not form a complete, equally spaced grid with at
        least two nodes per axis.
    """

    __slots__ = (
        "coefficient_file",
        "east_grid",
        "latitude_grid_deg",
        "latitude_max_deg",
        "latitude_min_deg",
        "latitude_step_deg",
        "longitude_grid_deg",
        "longitude_max_deg",
        "longitude_min_deg",
        "longitude_step_deg",
        "north_grid",
        "radial_grid",
    )

    def __init__(self, coefficient_file: str | Path) -> None:
        path = Path(coefficient_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Ocean pole-tide coefficient file not found: {path}")
        self.coefficient_file = path
        self._read(path)

    @staticmethod
    def _open(path: Path) -> TextIO:
        if path.suffix.lower() == ".gz":
            return gzip.open(path, "rt")
        return path.open("rt")

    def _read(self, path: Path) -> None:
        with self._open(path) as stream:
            records = np.fromregex(stream, _COEFFICIENT_ROW, dtype=[(str(i), float) for i in range(8)])
        values = records.view(float).reshape(-1, 8)
        if not len(values):
            raise ValueError(f"No numeric ocean pole-tide coefficients found in {path}.")
        longitude = values[:, 0]
        latitude = values[:, 1]
        longitude_grid, longitude_index = np.unique(longitude, return_inverse=True)
        latitude_grid, latitude_index = np.unique(latitude, return_inverse=True)

        n_latitude = latitude_grid.size
        n_longitude = longitude_grid.size
        if n_latitude < 2 or n_longitude < 2:
            raise ValueError("Ocean pole-tide grid needs at least two nodes per axis.")
        if values.shape[0] != n_latitude * n_longitude:
            raise ValueError(
                "Ocean pole-tide coefficient file is not a complete regular grid: "
                f"rows={values.shape[0]}, expected={n_latitude * n_longitude}."
            )

        longitude_step = float(np.median(np.diff(longitude_grid)))
        latitude_step = float(np.median(np.diff(latitude_grid)))
        if not np.allclose(np.diff(longitude_grid), longitude_step):
            raise ValueError("Ocean pole-tide longitude grid is not equally spaced.")
        if not np.allclose(np.diff(latitude_grid), latitude_step):
            raise ValueError("Ocean pole-tide latitude grid is not equally spaced.")

        radial = np.empty((n_latitude, n_longitude), dtype=np.complex128)
        north = np.empty_like(radial)
        east = np.empty_like(radial)
        for grid, column in ((radial, 2), (north, 4), (east, 6)):
            grid.real[latitude_index, longitude_index] = values[:, column]
            grid.imag[latitude_index, longitude_index] = values[:, column + 1]

        for array in (longitude_grid, latitude_grid, radial, north, east):
            array.setflags(write=False)

        self.longitude_grid_deg = longitude_grid
        self.latitude_grid_deg = latitude_grid
        self.radial_grid = radial
        self.north_grid = north
        self.east_grid = east
        self.longitude_step_deg = longitude_step
        self.latitude_step_deg = latitude_step
        self.longitude_min_deg = float(longitude_grid[0])
        self.longitude_max_deg = float(longitude_grid[-1])
        self.latitude_min_deg = float(latitude_grid[0])
        self.latitude_max_deg = float(latitude_grid[-1])

    def _wrap_longitude(self, longitude_deg: float) -> float:
        return ((float(longitude_deg) - self.longitude_min_deg) % 360.0) + self.longitude_min_deg

    def _interpolate(self, grid: np.ndarray, latitude_deg: float, longitude_deg: float) -> complex:
        latitude = float(np.clip(latitude_deg, self.latitude_min_deg, self.latitude_max_deg))
        longitude = self._wrap_longitude(longitude_deg)

        y = (latitude - self.latitude_min_deg) / self.latitude_step_deg
        i0 = int(np.floor(y))
        fraction_y = y - i0
        if i0 >= self.latitude_grid_deg.size - 1:
            i0 = self.latitude_grid_deg.size - 2
            fraction_y = 1.0
        elif i0 < 0:
            i0 = 0
            fraction_y = 0.0
        i1 = i0 + 1

        x = (longitude - self.longitude_min_deg) / self.longitude_step_deg
        floor_x = np.floor(x)
        j0 = int(floor_x) % self.longitude_grid_deg.size
        fraction_x = x - floor_x
        j1 = (j0 + 1) % self.longitude_grid_deg.size

        lower = (1.0 - fraction_x) * grid[i0, j0] + fraction_x * grid[i0, j1]
        upper = (1.0 - fraction_x) * grid[i1, j0] + fraction_x * grid[i1, j1]
        return complex((1.0 - fraction_y) * lower + fraction_y * upper)

    def coefficients_at(
        self,
        reference_position_itrf_m: ArrayLike,
    ) -> OceanPoleTideCoefficients:
        """Return the loading coefficients interpolated at a station.

        Parameters
        ----------
        reference_position_itrf_m : array_like, shape (3,)
            Station position (X, Y, Z) in the ITRF, in metres; converted to WGS84 geodetic coordinates
            for the interpolation.

        Returns
        -------
        OceanPoleTideCoefficients
            Station coordinates and interpolated coefficients.
        """
        site = itrf2geodetic(reference_position_itrf_m)
        latitude_deg = site.latitude_deg
        longitude_deg = site.longitude_deg
        return OceanPoleTideCoefficients(
            latitude_rad=site.latitude_rad,
            longitude_rad=site.longitude_rad,
            ellipsoidal_height_m=site.ellipsoidal_height_m,
            radial=self._interpolate(self.radial_grid, latitude_deg, longitude_deg),
            north=self._interpolate(self.north_grid, latitude_deg, longitude_deg),
            east=self._interpolate(self.east_grid, latitude_deg, longitude_deg),
        )


class Iers2010OceanPoleTide:
    """IERS 2010 ocean pole tide loading displacement of a station.

    Each local component (up, north, east) of the displacement is

        K [(m1 gamma_R + m2 gamma_I) u^R + (m2 gamma_R - m1 gamma_I) u^I],

    with the wobble variables m1, m2 in radians (relative to the 2018 secular pole), the coefficients
    u^R + i u^I interpolated from the grid, gamma = 0.6870 + 0.0036i, and
    K = 4 pi G a_E rho_w H_p / (3 g_e), H_p = sqrt(8 pi / 15) Omega^2 a_E^4 / GM.

    Parameters
    ----------
    grid : OceanPoleTideGrid
        Loading coefficient grid.
    earth_orientation_provider : EarthOrientationProvider
        Source of the polar motion.

    Attributes
    ----------
    scale_m : float
        Scale factor K, in metres.

    References
    ----------
    .. [1] Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010), IERS Technical Note 36,
       Sec. 7.1.5.
    .. [2] Desai, S. D. (2002). Observing the pole tide with satellite altimetry. Journal of Geophysical
       Research, 107(C11), 3186.
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            path(
                "coefficientFile",
                required=True,
                non_empty=True,
                description="Ocean pole-tide loading coefficient grid (IERS `opoleloadcoefcmcor.txt` format).",
            ),
        ),
        type_name="iers2010OceanPoleTide",
    )

    @classmethod
    def from_config(cls, cfg, ctx, *, earth_orientation_provider):
        """Create the model with the cached coefficient grid of ``coefficientFile``."""
        from lunarops.classes.displacement import (
            OceanPoleTideGrid,
        )
        from lunarops.classes.displacement.coefficients import coefficient_file as _coefficient_file

        return cls(
            grid=_coefficient_file(OceanPoleTideGrid, ctx.resolve_path(cfg["coefficientFile"])),
            earth_orientation_provider=earth_orientation_provider,
        )

    def __init__(
        self,
        grid: OceanPoleTideGrid,
        earth_orientation_provider: EarthOrientationProvider,
    ) -> None:
        pole_tide_height_m = (
            np.sqrt(8.0 * np.pi / 15.0)
            * _EARTH_ANGULAR_VELOCITY_RAD_S**2
            * _EARTH_EQUATORIAL_RADIUS_M**4
            / _EARTH_GM_M3_S2
        )
        scale_m = (
            4.0
            * np.pi
            * _GRAVITATIONAL_CONSTANT_M3_KG_S2
            * _EARTH_EQUATORIAL_RADIUS_M
            * _SEAWATER_DENSITY_KG_M3
            * pole_tide_height_m
            / (3.0 * _EQUATORIAL_GRAVITY_M_S2)
        )
        self.grid = grid
        self.earth_orientation_provider = earth_orientation_provider
        self.scale_m = float(scale_m)
        self._coefficient_cache: dict[tuple, OceanPoleTideCoefficients] = {}

    def evaluate(self, data: StationDisplacementInput) -> np.ndarray:
        """Return the ocean pole tide displacement (X, Y, Z) in the ITRF, shape (3,), in metres."""
        key = tuple(data.reference_position_itrf_m)
        if key not in self._coefficient_cache:
            self._coefficient_cache[key] = self.grid.coefficients_at(data.reference_position_itrf_m)
        coefficients = self._coefficient_cache[key]
        wobble = polar_wobble(data.epoch_utc, self.earth_orientation_provider)
        m1_rad = float(np.deg2rad(wobble[0] / 3600.0))
        m2_rad = float(np.deg2rad(wobble[1] / 3600.0))
        gamma_real = float(_LOAD_LOVE_COMBINATION.real)
        gamma_imag = float(_LOAD_LOVE_COMBINATION.imag)
        factor_real = m1_rad * gamma_real + m2_rad * gamma_imag
        factor_imag = m2_rad * gamma_real - m1_rad * gamma_imag

        def displacement(coefficient: complex) -> float:
            return self.scale_m * (factor_real * coefficient.real + factor_imag * coefficient.imag)

        enu_m = np.array(
            [
                displacement(coefficients.east),
                displacement(coefficients.north),
                displacement(coefficients.radial),
            ],
            dtype=float,
        )
        itrf_m = enu2itrf(
            enu_m,
            latitude_rad=coefficients.latitude_rad,
            longitude_rad=coefficients.longitude_rad,
        )
        return itrf_m

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        """Return the ocean pole tide displacement (X, Y, Z) in the ITRF, shape (3,), in metres."""
        return self.evaluate(data)
