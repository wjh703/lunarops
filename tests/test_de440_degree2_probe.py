"""Independent quadrupole torque check for production Pavlov static offsets."""

from types import SimpleNamespace

import numpy as np
import pytest

from lunarops.classes.dynamics.deformation import LunarDeformation, LunarDeformationParameters
from lunarops.classes.dynamics.gravity import GravityField
from lunarops.classes.dynamics.deformation import degree2_gravity_coefficients_from_inertia

C21 = 2.2779569194479859e-10
S21 = 1.3343450358392620e-9


@pytest.mark.parametrize("c,s,s22", [(C21, 0, 0), (0, S21, 0),
                                  (C21, S21, 0), (-C21, -S21, 0), (0, 0, 3e-9)])
def test_offset_field_torque_matches_independent_cartesian_quadrupole(c, s, s22):
    radius, moon_mu, source_mu = 1738000., 4.9028e12, 3.986e14
    inertia = np.diag([.39276, .39285, .39301])
    field = GravityField(degree2_gravity_coefficients_from_inertia(
        inertia, gravitational_parameter_m3_s2=moon_mu, radius_m=radius))
    location = np.array([3.7e8, -1.5e8, 2.2e7])
    before = field.nonspherical_accelerations(location[None])[0]
    deformation = LunarDeformation(
        LunarDeformationParameters(.02419, 12000, 2.66e-6, c21_static=c, s21_static=s, s22_static=s22),
        inertia, radius, source_mu, moon_mu, field,
    )
    result = deformation.evaluate(SimpleNamespace(), rigid=True)
    after = field.nonspherical_accelerations(location[None])[0]
    torque = np.cross(location, -source_mu / moon_mu * (after - before)) / radius**2
    delta = np.array([[0, -2*s22, -c], [-2*s22, 0, -s], [-c, -s, 0]])
    unit = location / np.linalg.norm(location)
    expected = 3 * source_mu / np.linalg.norm(location)**3 * np.cross(unit, delta @ unit)
    np.testing.assert_allclose(torque, expected, rtol=2e-9, atol=1e-30)
    np.testing.assert_array_equal(result.total, inertia)
    # Repeated deformation must replace, rather than accumulate, the offset.
    deformation.evaluate(SimpleNamespace(), rigid=True)
    np.testing.assert_array_equal(after, field.nonspherical_accelerations(location[None])[0])
