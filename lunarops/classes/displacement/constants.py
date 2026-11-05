"""Default constants of the lunar solid-tide displacement model.

``LUNAR_H2`` and ``LUNAR_L2`` are the dimensionless degree-2 lunar displacement Love numbers;
``MOON_REFERENCE_RADIUS_M`` is the lunar reference radius in metres.
"""

from __future__ import annotations

# Lunar tidal constants compatible with the DE430 lunar solution.
LUNAR_H2 = 0.0423
LUNAR_L2 = 0.0107
MOON_REFERENCE_RADIUS_M = 1_737_400.0

__all__ = ["LUNAR_H2", "LUNAR_L2", "MOON_REFERENCE_RADIUS_M"]
