"""Default constants of the lunar solid-tide displacement model.

``LUNAR_H2`` and ``LUNAR_L2`` are the dimensionless degree-2 lunar displacement Love numbers;
``MOON_REFERENCE_RADIUS_M`` is the lunar reference radius in metres. ``LUNAR_H3`` and ``LUNAR_L3`` are the
degree-3 Love numbers and ``LUNAR_H2_OVER_Q`` and ``LUNAR_L2_OVER_Q`` the dissipation factors of the JPL LLR model
(Williams & Boggs 2020, JPL IOM 335N-20-01, Sec. 3.12 and App. A1), not applied by default.
``LUNAR_TIDAL_DISSIPATION_COEFFICIENTS_M`` is Table A.1 of that memo: per reflector and Delaunay argument
(``l`` anomalistic month, ``F`` nodical month), the cosine and sine coefficients of the up, east and north
displacements in metres per unit dissipation factor; the up columns are multiplied by ``h2/Q_h`` and the east and
north columns by ``l2/Q_l``.
"""

from __future__ import annotations

# Lunar tidal constants compatible with the DE430 lunar solution.
LUNAR_H2 = 0.0423
LUNAR_L2 = 0.0107
MOON_REFERENCE_RADIUS_M = 1_737_400.0

# Williams & Boggs (2020), Sec. 3.12; DE440 model values.
LUNAR_H3 = 0.0234
LUNAR_L3 = 0.0030
LUNAR_H2_OVER_Q = 1.0e-3
LUNAR_L2_OVER_Q = 3.0e-5

# Williams & Boggs (2020), Table A.1: {reflector: {argument: (up cos, up sin, east cos, east sin, north cos,
# north sin)}}, metres per unit dissipation factor.
LUNAR_TIDAL_DISSIPATION_COEFFICIENTS_M = {
    "APOLLO11": {
        "l": (-1.540, 1.618, -2.924, -2.282, 0.037, -0.064),
        "F": (0.050, 0.000, -0.022, 0.000, 4.158, -0.010),
    },
    "APOLLO14": {"l": (1.229, 1.800, -3.464, 1.834, 0.156, 0.361), "F": (-0.273, 0.000, -0.086, 0.002, 4.289, 0.008)},
    "APOLLO15": {
        "l": (-0.204, 1.484, -3.790, -0.326, 0.206, -2.494),
        "F": (1.790, -0.001, -0.126, -0.011, 2.767, -0.001),
    },
    "LUNOKHOD1": {"l": (1.232, 0.240, -1.119, 2.338, -1.962, -2.048), "F": (1.806, 0.000, 1.614, -0.013, 0.856, 0.003)},
    "LUNOKHOD2": {
        "l": (-1.511, 0.843, -1.828, -2.491, 1.456, -1.841),
        "F": (1.527, 0.000, -1.015, -0.010, 2.411, -0.008),
    },
}

__all__ = [
    "LUNAR_H2",
    "LUNAR_H2_OVER_Q",
    "LUNAR_H3",
    "LUNAR_L2",
    "LUNAR_L2_OVER_Q",
    "LUNAR_L3",
    "LUNAR_TIDAL_DISSIPATION_COEFFICIENTS_M",
    "MOON_REFERENCE_RADIUS_M",
]
