"""Fundamental physical constants shared across LunarOps.

Constants that belong to one model (for example gravitational parameters of an ephemeris) are defined with
that model.
"""

from __future__ import annotations

import math

#: Speed of light in vacuum, in m/s (defining constant; IERS Conventions (2010), Table 1.1).
C = 299_792_458.0
#: Square of the speed of light, in m^2/s^2.
C2 = C * C
#: Length of one day, in SI seconds.
SECONDS_PER_DAY = 86400.0
#: Length of one hour, in seconds.
SECONDS_PER_HOUR = 3600.0
DAYS_PER_JULIAN_YEAR = 365.25
DAYS_PER_JULIAN_CENTURY = 36525.0
SECONDS_PER_JULIAN_YEAR = SECONDS_PER_DAY * DAYS_PER_JULIAN_YEAR
J2000_JD = 2451545.0
"""Julian date of the J2000.0 epoch (2000-01-01 12:00), in whichever scale the caller uses."""
MJD_ZERO_JD = 2400000.5
"""Julian date of the Modified Julian Date origin (MJD = JD - MJD_ZERO_JD)."""
J2000_MJD = 51544.5
"""Modified Julian Date of the J2000.0 epoch."""
ARCSEC_TO_RAD = 4.848136811095359935899141e-6
MAS_TO_RAD = math.radians(1.0 / 3_600_000.0)

__all__ = [
    "ARCSEC_TO_RAD",
    "C2",
    "DAYS_PER_JULIAN_CENTURY",
    "DAYS_PER_JULIAN_YEAR",
    "J2000_JD",
    "J2000_MJD",
    "MAS_TO_RAD",
    "MJD_ZERO_JD",
    "SECONDS_PER_DAY",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_JULIAN_YEAR",
    "C",
]
