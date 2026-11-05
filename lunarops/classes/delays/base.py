"""Input record of the optical tropospheric delay models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TroposphereInput:
    """Station meteorology and geometry for an optical tropospheric slant delay.

    Attributes
    ----------
    pressure_hpa : float
        Surface atmospheric pressure at the station, in hPa.
    temperature_k : float
        Surface temperature at the station, in kelvin.
    relative_humidity_percent : float
        Surface relative humidity, in percent (0-100).
    latitude_rad : float
        Geodetic latitude of the station, north positive, in radians.
    height_m : float
        Station height above the ellipsoid, in metres.
    wavelength_um : float
        Laser wavelength, in micrometres.
    """

    pressure_hpa: float
    temperature_k: float
    relative_humidity_percent: float
    latitude_rad: float
    height_m: float
    wavelength_um: float
