"""Independent numerical checks for the Pavlov/DE440 deformation kernels."""

import numpy as np
from lunarops.classes.dynamics.forces import EarthTideModel, EarthTideParameters
from lunarops.classes.dynamics.inertia import LunarInertiaModel, LunarInertiaParameters
from lunarops.classes.time import Epoch


def _epoch():
    return Epoch(2451545.0, 0.0, "tdb")


def test_pavlov_inertia_matches_independent_eq15_reference():
    p = LunarInertiaParameters(include_tidal_deformation=True, include_rotational_deformation=True)
    epoch = _epoch()
    states = np.array([[0.0, 0.0, 0.0, 0, 0, 0], [3.84e8, 1.2e7, -2.0e6, 0, 0, 0]], float)
    q = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    history = lambda names, t: states
    attitude = lambda t: q
    model = LunarInertiaModel(3.986004418e14, 4.9048695e12, history, attitude, parameters=p)
    got = model.evaluate(epoch)
    r = q.T @ (states[0, :3] - states[1, :3])
    u = r / np.linalg.norm(r)
    tide = -p.love_k2 * (model.earth_gm / model.moon_gm) * (p.radius_m / np.linalg.norm(r)) ** 3 * (
        np.outer(u, u) - np.eye(3) / 3
    )
    w = np.zeros(3)
    spin_tensor = np.outer(w, w) - np.eye(3) * (w @ w - p.mean_motion_rad_s**2) / 3
    spin_tensor[2, 2] -= p.mean_motion_rad_s**2
    spin = p.love_k2 * p.radius_m**3 / (3 * model.moon_gm) * spin_tensor
    reference = p.undistorted() + tide + spin
    np.testing.assert_allclose(got["normalized_inertia"], reference, rtol=0, atol=2e-18)


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
        3.986004418e14, 4.9048695e12, history, parameters=p,
        tide_raisers=("MOON",), tide_raiser_gm={"MOON": 4.9048695e12},
    )
    got = model.relative_acceleration(epoch, earth, moon)
    r = moon - earth
    s = moon - earth
    radius = np.linalg.norm(r)
    rho = np.array((r[0], r[1], 0.0)); z = r[2]
    rho0 = np.array((s[0], s[1], 0.0)); z0 = s[2]; norm0 = np.linalg.norm(s)
    common = r / radius**2
    t0 = (2*z*z0**2*np.array((0.,0.,1.)) + np.dot(rho0,rho0)*rho
          - 5*((z*z0)**2 + .5*np.dot(rho,rho)*np.dot(rho0,rho0))*common
          + norm0**2*r) / norm0**5
    factor = 1.5*(model.earth_gm + model.moon_gm)/model.earth_gm
    factor *= 4.9048695e12 * model.earth_radius_m**5 / radius**5
    reference = factor * p.k20 * t0
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-30)
