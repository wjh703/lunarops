"""Whole-Moon inertia fixtures with fully normalized degree-2 gravity."""

import numpy as np

from lunarops.classes.dynamics.deformation import _degree2_coefficients_from_inertia
from lunarops.classes.dynamics.gravity import GravityCoefficients


def inertia_gravity(inertia, *, gravitational_parameter_m3_s2, radius_m,
                    c21_static=0.0, s21_static=0.0, s22_static=0.0):
    coefficients = np.zeros((2, 3, 3))
    coefficients[0, 0, 0] = 1.0
    coefficients[:, 2, :3] = _degree2_coefficients_from_inertia(
        inertia, c21_static=c21_static, s21_static=s21_static, s22_static=s22_static,
    )
    return GravityCoefficients(coefficients, gravitational_parameter_m3_s2, radius_m)
