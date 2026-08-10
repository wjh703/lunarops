from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import ArrayLike

from lunarops.classes.time import Epoch


class GravitationalDelay(ABC):
    """Interface for a one-way gravitational path delay model."""

    @abstractmethod
    def path_delay_m(
        self,
        transmitter_bcrs_m: ArrayLike,
        receiver_bcrs_m: ArrayLike,
        epoch_tdb: Epoch,
    ) -> float:
        """Return the one-way equivalent path delay in meters."""


class ZeroGravitationalDelay(GravitationalDelay):
    """Gravitational delay model that always returns zero."""

    def path_delay_m(self, transmitter_bcrs_m, receiver_bcrs_m, epoch_tdb: Epoch) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class TroposphereInput:
    """Inputs required to evaluate an optical tropospheric slant delay."""

    elevation_rad: float
    pressure_hpa: float
    temperature_k: float
    relative_humidity_percent: float
    latitude_rad: float
    height_m: float
    wavelength_um: float

    def __post_init__(self) -> None:
        for name in (
            "elevation_rad",
            "pressure_hpa",
            "temperature_k",
            "relative_humidity_percent",
            "latitude_rad",
            "height_m",
            "wavelength_um",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if not -0.5 * math.pi <= self.elevation_rad <= 0.5 * math.pi:
            raise ValueError("elevation_rad must be in [-pi/2, pi/2].")
        if self.pressure_hpa <= 0.0:
            raise ValueError("pressure_hpa must be positive.")
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive.")
        if not 0.0 <= self.relative_humidity_percent <= 100.0:
            raise ValueError("relative_humidity_percent must be in [0, 100].")
        if not -0.5 * math.pi <= self.latitude_rad <= 0.5 * math.pi:
            raise ValueError("latitude_rad must be in [-pi/2, pi/2].")
        if self.wavelength_um <= 0.0:
            raise ValueError("wavelength_um must be positive.")


class TroposphereDelay(ABC):
    """Interface for a one-way tropospheric slant-delay model."""

    @property
    def elevation_floor_rad(self) -> float | None:
        """Lowest elevation used by the model, or ``None`` if unclamped."""
        return None

    @abstractmethod
    def slant_delay_m(self, data: TroposphereInput) -> float:
        """Return the one-way tropospheric slant delay in meters."""


class ZeroTroposphereDelay(TroposphereDelay):
    """Tropospheric delay model that always returns zero."""

    def slant_delay_m(self, data: TroposphereInput) -> float:
        return 0.0
