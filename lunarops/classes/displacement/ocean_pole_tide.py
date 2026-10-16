"""IERS 2010 ocean pole-tide loading."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import ArrayLike

from lunarops.classes.frames.earth_orientation import EarthOrientationProvider

from .base import StationDisplacementInput
from .pole_tide import PolarWobble, polar_wobble
from .terrestrial_geometry import enu2itrf, itrf2geodetic

_GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67428e-11
_EARTH_EQUATORIAL_RADIUS_M = 6_378_136.6
_EARTH_GM_M3_S2 = 3.986004418e14
_EARTH_ANGULAR_VELOCITY_RAD_S = 7.292115e-5
_EQUATORIAL_GRAVITY_M_S2 = 9.7803278
_SEAWATER_DENSITY_KG_M3 = 1025.0
_LOAD_LOVE_COMBINATION = 0.6870 + 0.0036j


@dataclass(frozen=True, slots=True)
class OceanPoleTideCoefficients:
    """Interpolated complex loading coefficients at one station."""

    latitude_rad: float
    longitude_rad: float
    ellipsoidal_height_m: float
    radial: complex
    north: complex
    east: complex


@dataclass(frozen=True, slots=True)
class OceanPoleTideGridInfo:
    coefficient_file: Path
    latitude_nodes: int
    longitude_nodes: int
    latitude_min_deg: float
    latitude_max_deg: float
    longitude_min_deg: float
    longitude_max_deg: float
    latitude_step_deg: float
    longitude_step_deg: float


@dataclass(frozen=True, slots=True, eq=False)
class OceanPoleTideResult:
    displacement_itrf_m: np.ndarray
    displacement_enu_m: np.ndarray
    coefficients: OceanPoleTideCoefficients
    wobble: PolarWobble


class OceanPoleTideGrid:
    """Regular complex loading-coefficient grid with bilinear interpolation."""

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
        rows: list[list[float]] = []
        with self._open(path) as stream:
            for line_number, line in enumerate(stream, start=1):
                text = line.strip()
                if not text or text.startswith(("#", "%", "!")):
                    continue
                parts = text.split()
                if len(parts) < 8:
                    continue
                try:
                    rows.append([float(value) for value in parts[:8]])
                except ValueError:
                    # Published grids may contain unmarked textual headers.
                    continue

        if not rows:
            raise ValueError(f"No numeric ocean pole-tide coefficients found in {path}.")

        values = np.asarray(rows, dtype=float)
        longitude = values[:, 0]
        latitude = values[:, 1]
        longitude_grid = np.unique(longitude)
        latitude_grid = np.unique(latitude)
        longitude_grid.sort()
        latitude_grid.sort()

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
        longitude_index = {float(value): index for index, value in enumerate(longitude_grid)}
        latitude_index = {float(value): index for index, value in enumerate(latitude_grid)}

        for row in values:
            lon_deg, lat_deg = float(row[0]), float(row[1])
            i = latitude_index[lat_deg]
            j = longitude_index[lon_deg]
            radial[i, j] = complex(row[2], row[3])
            north[i, j] = complex(row[4], row[5])
            east[i, j] = complex(row[6], row[7])

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

    @property
    def info(self) -> OceanPoleTideGridInfo:
        return OceanPoleTideGridInfo(
            coefficient_file=self.coefficient_file,
            latitude_nodes=int(self.latitude_grid_deg.size),
            longitude_nodes=int(self.longitude_grid_deg.size),
            latitude_min_deg=self.latitude_min_deg,
            latitude_max_deg=self.latitude_max_deg,
            longitude_min_deg=self.longitude_min_deg,
            longitude_max_deg=self.longitude_max_deg,
            latitude_step_deg=self.latitude_step_deg,
            longitude_step_deg=self.longitude_step_deg,
        )

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
    """Ocean pole-tide loading using an explicit grid and IERS table."""

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

    def evaluate(self, data: StationDisplacementInput) -> OceanPoleTideResult:
        coefficients = self.grid.coefficients_at(data.reference_position_itrf_m)
        wobble = polar_wobble(data.epoch_utc, self.earth_orientation_provider)
        gamma_real = float(_LOAD_LOVE_COMBINATION.real)
        gamma_imag = float(_LOAD_LOVE_COMBINATION.imag)
        factor_real = wobble.m1_rad * gamma_real + wobble.m2_rad * gamma_imag
        factor_imag = wobble.m2_rad * gamma_real - wobble.m1_rad * gamma_imag

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
        return OceanPoleTideResult(
            displacement_itrf_m=itrf_m,
            displacement_enu_m=enu_m,
            coefficients=coefficients,
            wobble=wobble,
        )

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        return self.evaluate(data).displacement_itrf_m
