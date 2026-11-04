"""IERS 2010 Eq. (11.17) gravitational path delay."""

import numpy as np

from lunarops.base.constants import C2
from lunarops.classes.delays.shapiro import Iers2010ShapiroDelay
from lunarops.classes.frames.coordinate_scaling import GM_BY_BODY
from tests.support.synthetic import J2000_TDB, LinearEphemeris
from tests.tolerances import REL, Tolerance, assert_close

AU_M = 1.495978707e11
EARTH_MOON_M = 3.844e8
FAR_M = 1.0e22  # bodies this far away contribute below 1e-12 m


def _delay(sun_position):
    bodies = {"SUN": (sun_position, (0, 0, 0))}
    for k, name in enumerate(("EARTH", "MOON", "JUPITER BARYCENTER", "SATURN BARYCENTER"), start=1):
        bodies[name] = ((FAR_M * k, FAR_M, 0.0), (0, 0, 0))
    model = Iers2010ShapiroDelay(body_states=LinearEphemeris(bodies).body_state_matrix_bcrs)
    transmitter, receiver = np.zeros(3), np.array([0.0, EARTH_MOON_M, 0.0])
    return model.path_delay_m(transmitter, receiver, J2000_TDB), transmitter, receiver


def test_solar_delay_follows_the_logarithmic_formula():
    sun = np.array([AU_M, 0.0, 0.0])
    delay, transmitter, receiver = _delay(sun)

    r1, r2 = np.linalg.norm(transmitter - sun), np.linalg.norm(receiver - sun)
    rho = np.linalg.norm(receiver - transmitter)
    assert_close(delay, 2 * GM_BY_BODY["SUN"] / C2 * np.log((r1 + r2 + rho) / (r1 + r2 - rho)), REL)


def test_solar_delay_of_an_earth_moon_path_is_about_seven_metres():
    delay, _, _ = _delay(np.array([AU_M, 0.0, 0.0]))
    # For rho << R the logarithm is rho/R to second order in rho/R.
    weak_field = 2 * GM_BY_BODY["SUN"] / C2 * EARTH_MOON_M / AU_M
    assert_close(delay, weak_field, Tolerance("WEAK_FIELD_EXPANSION", rtol=1e-5))
    assert 7.5 < delay < 7.7


def test_delay_grows_when_the_path_passes_closer_to_the_sun():
    far, _, _ = _delay(np.array([AU_M, 0.0, 0.0]))
    near, _, _ = _delay(np.array([0.1 * AU_M, 0.5 * EARTH_MOON_M, 0.0]))
    assert near > far > 0.0
