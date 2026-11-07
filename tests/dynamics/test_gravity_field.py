"""The native spherical-harmonic kernel against an independent NumPy potential."""

from math import factorial

import numpy as np
import pytest
from scipy.special import lpmv

from lunarops.classes.dynamics import make_gravity_coefficients
from lunarops.classes.dynamics.gravity import GravityField
from tests.support.dynamics import MOON_MU, MOON_RADIUS, moon_field
from tests.tolerances import REL, Tolerance, assert_vector_close


def _normalized_legendre(n, m, x):
    """Fully normalized associated Legendre function without the Condon-Shortley phase."""
    norm = np.sqrt((1 if m == 0 else 2) * (2 * n + 1) * factorial(n - m) / factorial(n + m))
    return norm * (-1) ** m * lpmv(m, n, x)


def _nonspherical_potential(field: GravityField, position):
    """V = GM/r sum_{n>=2} (R/r)^n sum_m Pbar_nm(sin phi) (Cbar_nm cos m lambda + Sbar_nm sin m lambda)."""
    coefficients = field.coefficients
    cosine, sine = coefficients.coeffs[0], coefficients.coeffs[1]
    x, y, z = position
    r = np.linalg.norm(position)
    latitude_sine, longitude = z / r, np.arctan2(y, x)
    total = 0.0
    for n in range(2, cosine.shape[0]):
        for m in range(n + 1):
            harmonic = cosine[n, m] * np.cos(m * longitude) + sine[n, m] * np.sin(m * longitude)
            total += (coefficients.radius_m / r) ** n * _normalized_legendre(n, m, latitude_sine) * harmonic
    return coefficients.gm_m3_s2 / r * total


@pytest.mark.parametrize(
    "position",
    [
        np.array([3.1e6, -1.2e6, 2.4e6]),
        np.array([-2.0e6, 4.0e6, -1.5e6]),
        np.array([1.0e7, 2.0e6, 9.0e6]),
    ],
)
def test_acceleration_is_the_gradient_of_the_harmonic_potential(position):
    field = moon_field()
    h = 1e-6 * np.linalg.norm(position)
    gradient = np.array(
        [
            (_nonspherical_potential(field, position + h * e) - _nonspherical_potential(field, position - h * e))
            / (2 * h)
            for e in np.eye(3)
        ]
    )
    # Central differences with h = 1e-6 r: truncation 1e-12 and round off about 1e-9 relative.
    assert_vector_close(
        field.nonspherical_accelerations(position[None])[0], gradient, Tolerance("POTENTIAL_GRADIENT", rtol=1e-8)
    )


def test_zonal_c20_gives_the_closed_form_j2_acceleration():
    j2 = 2.03e-4
    cosine = np.zeros((3, 3))
    cosine[0, 0], cosine[2, 0] = 1.0, -j2 / np.sqrt(5.0)
    field = GravityField(make_gravity_coefficients(cosine, np.zeros((3, 3)), gm_m3_s2=MOON_MU, radius_m=MOON_RADIUS))
    positions = np.array([[2.0e6, -1.0e6, 1.5e6], [0.0, 0.0, 3.0e6], [4.0e6, 0.0, 0.0], [1e8, 3e8, -2e8]])
    r = np.linalg.norm(positions, axis=1)[:, None]
    z2 = (positions[:, 2:3] / r) ** 2
    expected = -1.5 * j2 * MOON_MU * MOON_RADIUS**2 / r**5 * positions * np.hstack([1 - 5 * z2, 1 - 5 * z2, 3 - 5 * z2])

    assert_vector_close(field.nonspherical_accelerations(positions), expected, REL)
