"""Fundamental physical constants shared across LunarOps.

Constants that belong to one model (for example gravitational parameters of an ephemeris) are defined with
that model.
"""

from __future__ import annotations

#: Speed of light in vacuum, in m/s (defining constant; IERS Conventions (2010), Table 1.1).
C = 299_792_458.0
#: Square of the speed of light, in m^2/s^2.
C2 = C * C
#: Length of one day, in SI seconds.
SECONDS_PER_DAY = 86400.0

__all__ = ["C2", "SECONDS_PER_DAY", "C"]
