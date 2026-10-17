from lunarops.classes.dynamics.deformation import degree2_gravity_coefficients_from_inertia

"""Independent numerical checks for the Pavlov/DE440 deformation kernels."""

import numpy as np
import pytest

from lunarops.classes.dynamics.forces import EarthTideModel, EarthTideParameters
from lunarops.classes.dynamics.gravity import GravityCoefficients, GravityField
from lunarops.classes.dynamics.inertia import (
    LunarDegree2GravityModel,
    LunarDegree2GravityParameters,
)
from lunarops.classes.time import Epoch


def _epoch():
    return Epoch(2451545.0, 0.0, "tdb")


def test_pavlov_inertia_matches_independent_eq15_reference():
    p = LunarDegree2GravityParameters(include_tidal_deformation=True, include_rotational_deformation=True)
    epoch = _epoch()
    states = np.array([[0.0, 0.0, 0.0, 0, 0, 0], [3.84e8, 1.2e7, -2.0e6, 0, 0, 0]], float)
    q = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    relative_position = lambda t: states[0, :3] - states[1, :3]
    attitude = lambda t: q
    model = LunarDegree2GravityModel(3.986004418e14, 4.9048695e12, relative_position, attitude, angular_velocity_provider=lambda t: np.zeros(3), parameters=p)
    got = model._evaluate_components(epoch)[0]
    r = q.T @ (states[0, :3] - states[1, :3])
    u = r / np.linalg.norm(r)
    tide = -p.love_k2 * (model.earth_gravitational_parameter_m3_s2 / model.moon_gravitational_parameter_m3_s2) * (p.radius_m / np.linalg.norm(r)) ** 3 * (
        np.outer(u, u) - np.eye(3) / 3
    )
    w = np.zeros(3)
    spin_tensor = np.outer(w, w) - np.eye(3) * (w @ w - p.mean_motion_rad_s**2) / 3
    spin_tensor[2, 2] -= p.mean_motion_rad_s**2
    spin = p.love_k2 * p.radius_m**3 / (3 * model.moon_gravitational_parameter_m3_s2) * spin_tensor
    reference = p.undistorted_inertia() + tide + spin
    np.testing.assert_allclose(got, reference, rtol=0, atol=2e-18)


@pytest.mark.parametrize("c21,s21,s22", [(0., 0., 0.), (2.2779569194479859e-10, 1.3343450358392620e-9, -3e-9)])
def test_pavlov_eq16_matches_independent_unnormalized_mapping(c21, s21, s22):
    inertia = np.array([[.39276, 1e-8, -2e-8], [1e-8, .39285, 3e-8], [-2e-8, 3e-8, .39301]])
    before = inertia.copy()
    result = degree2_gravity_coefficients_from_inertia(
        inertia, gravitational_parameter_m3_s2=4.9028e12, radius_m=1738000.,
        c21_static=c21, s21_static=s21, s22_static=s22,
    )
    # Independent Eq. (16), then convert unnormalized coefficients to 4pi.
    expected = np.array([
        [(inertia[0, 0] + inertia[1, 1] - 2*inertia[2, 2]) / (2*np.sqrt(5)),
         (c21 - inertia[0, 2]) / np.sqrt(5/3),
         (inertia[1, 1] - inertia[0, 0]) / (4*np.sqrt(5/12))],
        [0., (s21 - inertia[1, 2]) / np.sqrt(5/3),
         (s22 - inertia[0, 1]/2) / np.sqrt(5/12)],
    ])
    np.testing.assert_allclose(result.coeffs[:, 2, :3], expected, rtol=2e-12, atol=1e-20)
    np.testing.assert_array_equal(inertia, before)


def test_prescribed_pavlov_offsets_replace_static_field_without_double_counting():
    p = LunarDegree2GravityParameters(c21_static=2.2779569194479859e-10,
                                     s21_static=1.3343450358392620e-9, s22_static=4e-9)
    model = LunarDegree2GravityModel(3.986e14, 4.9028e12,
                                    lambda t: np.array([3.8e8, 2e7, -1e7]), lambda t: np.eye(3), angular_velocity_provider=lambda t: np.zeros(3), parameters=p)
    result = model._evaluate_components(_epoch())[0]
    zero_parameters = LunarDegree2GravityParameters()
    zero_model = LunarDegree2GravityModel(3.986e14, 4.9028e12,
                                         model.earth_minus_moon_position_provider, model.attitude_provider,
                                         angular_velocity_provider=lambda t: np.zeros(3), parameters=zero_parameters)
    zero_result = zero_model._evaluate_components(_epoch())[0]
    np.testing.assert_array_equal(result, zero_result)
    result_coefficients = degree2_gravity_coefficients_from_inertia(
        result, gravitational_parameter_m3_s2=4.9028e12, radius_m=p.radius_m,
        c21_static=p.c21_static, s21_static=p.s21_static, s22_static=p.s22_static,
    ).coeffs
    zero_coefficients = degree2_gravity_coefficients_from_inertia(
        zero_result, gravitational_parameter_m3_s2=4.9028e12, radius_m=p.radius_m,
    ).coeffs
    expected_offset = np.array([[0., p.c21_static / np.sqrt(5/3), 0.],
                               [0., p.s21_static / np.sqrt(5/3), p.s22_static / np.sqrt(5/12)]])
    np.testing.assert_allclose(result_coefficients[:, 2, :3],
                               zero_coefficients[:, 2, :3] + expected_offset,
                               rtol=0, atol=1e-22)
    # Include nonzero static offsets and a higher-degree coefficient in the GFC input.
    coefficients = np.zeros((2, 4, 4))
    coefficients[:, :3, :3] = result_coefficients
    coefficients[0, 2, 1] += 2e-9
    coefficients[0, 3, 1] = 1e-5
    static = GravityCoefficients(coefficients, 4.9028e12, p.radius_m)
    field = GravityField(static)
    model.update_gravity_field(_epoch(), field)
    np.testing.assert_allclose(field.coefficients.coeffs[:, 2, :3], result_coefficients[:, 2, :3], rtol=0, atol=1e-22)
    np.testing.assert_array_equal(field.coefficients.coeffs[:, 3], static.coeffs[:, 3])
    before = field.coefficients.coeffs.copy()
    model.update_gravity_field(_epoch(), field)
    np.testing.assert_array_equal(field.coefficients.coeffs, before)


def test_earth_tide_eq7_independent_single_raiser_kernel():
    p = EarthTideParameters(k21=0.0, k22=0.0)
    epoch = _epoch()
    earth = np.zeros(3)
    moon = np.array([3.84e8, 2.1e7, 1.1e7])
    sun = np.array([1.49e11, -2.0e10, 3.0e9])
    states = {"EARTH": earth, "MOON": moon, "SUN": sun}
    def history(names, t):
        return np.array([[*states[n], 0, 0, 0] for n in names], float)
    model = EarthTideModel(
        3.986004418e14, 4.9048695e12, tide_parameters=p,
        tide_raisers=("MOON",), tide_raiser_gm={"MOON": 4.9048695e12},
    )
    got = model.relative_acceleration(epoch, earth, moon, np.eye(3), history)
    r = moon - earth
    s = moon - earth
    radius = np.linalg.norm(r)
    rho = np.array((r[0], r[1], 0.0)); z = r[2]
    rho0 = np.array((s[0], s[1], 0.0)); z0 = s[2]; norm0 = np.linalg.norm(s)
    common = r / radius**2
    t0 = (2*z*z0**2*np.array((0.,0.,1.)) + np.dot(rho0,rho0)*rho
          - 5*((z*z0)**2 + .5*np.dot(rho,rho)*np.dot(rho0,rho0))*common
          + norm0**2*r) / norm0**5
    factor = 1.5*(model.earth_gravitational_parameter_m3_s2 + model.moon_gravitational_parameter_m3_s2)/model.earth_gravitational_parameter_m3_s2
    factor *= 4.9048695e12 * model.earth_radius_m**5 / radius**5
    reference = factor * p.k20 * t0
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-30)
