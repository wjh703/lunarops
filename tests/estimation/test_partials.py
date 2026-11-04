"""Analytic observation partials against central differences of the full observation model.

Partials are derivatives of the one-way computed range, so d(O - C)/dp = -partial.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lunarops.classes.ephemerides import FixedLunarOrientation, LunarRelativisticScale
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.observation import NptDataset, ObservationOptions
from tests.support.observation import (
    STATIC_BODIES,
    ZeroEarthOrientation,
    npt_record,
    observation_processor,
    static_frames,
    station_facing_the_moon,
)
from tests.support.synthetic import LinearEphemeris
from tests.tolerances import Tolerance, assert_close

ALL_PARTIALS = ObservationOptions(min_elevation_deg=-90.0, include_reflector_position_partials=True,
                                  include_station_position_partials=True, include_lunar_orientation_partials=True)
# The analytic partials ignore how the event epochs move with the parameter; the rotating station
# makes that a |v|/c ~ 1.6e-6 effect. Light-time solutions carry about 4e-7 m of iteration round off,
# so the steps move the reflector or station by metres (1 m, 1e-6 rad).
POSITION_PARTIAL = Tolerance("POSITION_PARTIAL_V_OVER_C", rtol=5e-6, atol=1e-6)
ORIENTATION_PARTIAL = Tolerance("ORIENTATION_PARTIAL_V_OVER_C", rtol=5e-6, atol=1.0, unit="m/rad")


def _frames(attitude=None):
    attitude = np.eye(3) if attitude is None else attitude
    return ReferenceFrameSystem(LinearEphemeris(STATIC_BODIES), ZeroEarthOrientation(),
                                FixedLunarOrientation(lambda epoch: attitude),
                                LunarRelativisticScale.from_convention("alreadyScaled"))


STATION = station_facing_the_moon(static_frames(), zenith_angle_deg=35.0)


def _oc(*, frames=None, station=STATION, reflector=None):
    processor = observation_processor(frames=frames or _frames(), station_itrf_m=station)
    if reflector is not None:
        record = processor.model_state.reflector_catalog["APOLLO15"]
        processor.model_state.reflector_catalog["APOLLO15"] = replace(record, moon_fixed_xyz_m=tuple(reflector))
    return processor.equations(NptDataset([npt_record()]), options=ALL_PARTIALS)


@pytest.fixture(scope="module")
def nominal():
    return _oc()


@pytest.mark.parametrize("axis", range(3))
def test_station_position_partial_matches_moving_the_station(nominal, axis):
    h = 1.0
    plus, minus = (_oc(station=STATION + sign * h * np.eye(3)[axis]) for sign in (1, -1))
    assert_close(-(plus.oc - minus.oc) / (2 * h), nominal.station_position_itrf[:, axis], POSITION_PARTIAL)


@pytest.mark.parametrize("axis", range(3))
def test_reflector_position_partial_matches_moving_the_reflector(nominal, axis):
    h = 1.0
    reference = np.array([1_737_400.0, 0.0, 0.0])
    plus, minus = (_oc(reflector=reference + sign * h * np.eye(3)[axis]) for sign in (1, -1))
    assert_close(-(plus.oc - minus.oc) / (2 * h), nominal.reflector_position_pa[:, axis], POSITION_PARTIAL)


@pytest.mark.parametrize("axis", range(3))
def test_lunar_orientation_partial_matches_rotating_the_principal_axes(nominal, axis):
    # A rotation by eps about PA axis a moves the reflector by eps (a x r).
    eps = 1.0e-6
    plus, minus = (_oc(frames=_frames(Rotation.from_rotvec(sign * eps * np.eye(3)[axis]).as_matrix()))
                   for sign in (1, -1))
    assert_close(-(plus.oc - minus.oc) / (2 * eps), nominal.lunar_orientation_pa[:, axis], ORIENTATION_PARTIAL)
