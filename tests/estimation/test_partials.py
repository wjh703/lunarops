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


class _OffsetEarthOrientation(ZeroEarthOrientation):
    """Constant polar motion and UT1 - UTC offsets."""

    def __init__(self, xp_arcsec=0.0, yp_arcsec=0.0, ut1_s=0.0):
        self.values = (xp_arcsec, yp_arcsec, ut1_s)

    def interpolate(self, epoch_utc):
        from lunarops.classes.frames.earth_orientation import EarthOrientationValues

        return EarthOrientationValues(*self.values)


@pytest.mark.parametrize(("component", "step"), [(0, 1.0), (1, 10.0), (2, 10.0)])
def test_earth_orientation_partials_match_changing_the_tabulated_values(component, step):
    """dUT1 (ms), dx_p and dy_p (mas) partials against central differences of the observation model."""
    from lunarops.estimation.earth_orientation_effects import earth_orientation_partials

    def frames(sign):
        offsets = np.zeros(3)
        offsets[component] = sign * step
        provider = _OffsetEarthOrientation(xp_arcsec=offsets[1] * 1e-3, yp_arcsec=offsets[2] * 1e-3,
                                           ut1_s=offsets[0] * 1e-3)
        return ReferenceFrameSystem(LinearEphemeris(STATIC_BODIES), provider,
                                    FixedLunarOrientation(lambda epoch: np.eye(3)),
                                    LunarRelativisticScale.from_convention("alreadyScaled"))

    # A station off the equator, so that every component has a lever arm.
    station = np.array([4_581_692.0, 556_196.0, 4_389_355.0])
    plus, minus = (_oc(frames=frames(sign), station=station) for sign in (1, -1))
    nominal = _oc(frames=frames(0), station=station)
    analytic = earth_orientation_partials(nominal.station_position_itrf, station[None, :])[:, component]
    numeric = -(plus.oc - minus.oc) / (2 * step)
    assert abs(analytic[0]) > 1.0e-3
    assert_close(numeric, analytic, Tolerance("EOP_PARTIAL_V_OVER_C", rtol=2e-5, atol=1e-6, unit="m/unit"))
