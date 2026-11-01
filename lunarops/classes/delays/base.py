from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TroposphereInput:
    """Inputs required to evaluate an optical tropospheric slant delay."""

    pressure_hpa: float
    temperature_k: float
    relative_humidity_percent: float
    latitude_rad: float
    height_m: float
    wavelength_um: float
