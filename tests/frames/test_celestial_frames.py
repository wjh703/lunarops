"""Body identifiers, lunar orientation and BCRS/GCRS/LCRS transforms."""

import numpy as np
import pytest

from lunarops.classes.ephemerides import (
    BodyId,
    FixedLunarOrientation,
    LunarRelativisticScale,
    canonical_body_id,
    naif_id,
)
from lunarops.classes.ephemerides.body_ids import BODY_BY_NAIF_ID, NAIF_ID_BY_BODY
from lunarops.classes.ephemerides.lunar_orientation import (
    EulerAngleLunarOrientation,
    LongitudeLibrationCorrection,
    normalize_longitude_libration_correction,
)
from lunarops.classes.frames import EarthOrientationSample, ReferenceFrameSystem, TabulatedEarthOrientation
from lunarops.classes.frames.coordinate_scaling import (
    GM_SUN,
    LunarRelativisticScaleConvention,
    normalize_lunar_relativistic_scale_convention,
)
from lunarops.classes.frames.relativistic import RelativisticFrameTransform
from lunarops.classes.time import Epoch, TimeScale
from tests.support.synthetic import LinearEphemeris
from tests.tolerances import REL, Tolerance, assert_close

EPOCH = Epoch(2451545.0, 0.0, TimeScale.TDB)
QUARTER_TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# Round trips pass through BCRS coordinates near 2e11 m, whose ulp is 3e-5 m.
ROUND_TRIP_M = Tolerance("BCRS_ROUND_TRIP_M", atol=1e-4, unit="m")


def _ephemeris():
    return LinearEphemeris(
        {
            "SUN": ([1.0e11, 1.0e9, 0.0], [0.0, 0.0, 0.0]),
            "EARTH": ([2.0e11, 2.0e9, 0.0], [0.0, 0.0, 0.0]),
            "MOON": ([3.0e11, 3.0e9, 0.0], [12_000.0, -18_000.0, 3_000.0]),
            **{
                name: ([k * 1.0e11, k * 1.0e9, 0.0], [0.0, 0.0, 0.0])
                for k, name in enumerate(
                    (
                        "MERCURY BARYCENTER",
                        "VENUS BARYCENTER",
                        "MARS BARYCENTER",
                        "JUPITER BARYCENTER",
                        "SATURN BARYCENTER",
                        "URANUS BARYCENTER",
                        "NEPTUNE BARYCENTER",
                    ),
                    start=4,
                )
            },
        }
    )


def test_body_identifiers_distinguish_centres_from_system_barycentres():
    assert canonical_body_id("SOLAR SYSTEM BARYCENTER") is BodyId.SOLAR_SYSTEM_BARYCENTER
    assert canonical_body_id("EARTH MOON BARYCENTER") is BodyId.EARTH_MOON_BARYCENTER
    assert (naif_id("MARS"), naif_id("MARS BARYCENTER"), naif_id("EARTH"), naif_id("EARTH MOON BARYCENTER")) == (
        499,
        4,
        399,
        3,
    )
    assert NAIF_ID_BY_BODY[BodyId.JUPITER_BARYCENTER] == 5
    assert BODY_BY_NAIF_ID[599] is BodyId.JUPITER
    for alias in ("SSB", "EMB", "EARTH BARYCENTER", "moon", " MOON"):
        with pytest.raises(ValueError):
            canonical_body_id(alias)


def test_relativistic_transforms_are_inverse_and_the_frame_system_wires_them():
    ephemeris = _ephemeris()
    eop = TabulatedEarthOrientation((EarthOrientationSample(51544.5, 0.1, 0.2, 0.0),))
    system = ReferenceFrameSystem(
        ephemeris,
        eop,
        FixedLunarOrientation(lambda _: QUARTER_TURN),
        LunarRelativisticScale.from_convention("alreadyScaled"),
    )
    transform = system.relativistic_transform
    gcrs = np.array([6_378_137.0, 100.0, -50.0])
    lcrs = np.array([1_737_400.0, -200.0, 75.0])
    delta = np.array([20.0, -10.0, 5.0])

    lunar_bcrs = transform.lcrs2bcrs(lcrs, EPOCH)

    assert_close(transform.bcrs2gcrs(transform.gcrs2bcrs(gcrs, EPOCH), EPOCH), gcrs, ROUND_TRIP_M)
    assert_close(transform.bcrs2lcrs(lunar_bcrs, EPOCH), lcrs, ROUND_TRIP_M)
    assert_close(
        transform.bcrs_vector2lcrs(transform.lcrs2bcrs(lcrs + delta, EPOCH) - lunar_bcrs, EPOCH), delta, ROUND_TRIP_M
    )
    assert_close(system.lunar_orientation.pa_to_lcrs_matrix(EPOCH) @ [1.0, 2.0, 3.0], [-2.0, 1.0, 3.0], REL)
    assert system.ephemeris is ephemeris


def test_external_potential_is_the_sum_of_gm_over_distance_without_duplicates():
    transform = RelativisticFrameTransform(_ephemeris(), LunarRelativisticScale.from_convention("alreadyScaled"))
    distance = np.linalg.norm(np.array([1.0e11, 1.0e9, 0.0]))

    potential = transform.external_gravitational_potential_m2_s2("EARTH", EPOCH, ("SUN",))

    assert_close(potential, GM_SUN / distance, REL)
    assert transform.external_gravitational_potential_m2_s2("EARTH", EPOCH, ("SUN", "SUN")) == potential
    with pytest.raises(ValueError, match="must not also appear"):
        transform.external_gravitational_potential_m2_s2("EARTH", EPOCH, ("EARTH",))


def test_orientation_and_scale_conventions_are_explicit():
    assert normalize_longitude_libration_correction(None) is LongitudeLibrationCorrection.NONE
    assert normalize_longitude_libration_correction(" INPOP21A ") is LongitudeLibrationCorrection.INPOP21A
    for legacy in (False, True, "", "no", "off", "false", "0"):
        with pytest.raises((TypeError, ValueError)):
            normalize_longitude_libration_correction(legacy)  # type: ignore[arg-type]
    assert FixedLunarOrientation(lambda _: np.eye(3)).longitude_libration_correction_rad(EPOCH) == 0.0
    assert (
        normalize_lunar_relativistic_scale_convention(" tdbCompatibleLunarSurface ")
        is LunarRelativisticScaleConvention.TDB_COMPATIBLE_LUNAR_SURFACE
    )
    assert (
        normalize_lunar_relativistic_scale_convention("alreadyscaled")
        is LunarRelativisticScaleConvention.ALREADY_SCALED
    )
    for name in ("inferredFromFilename", "de440"):
        with pytest.raises(ValueError, match="expected one of"):
            normalize_lunar_relativistic_scale_convention(name)


@pytest.mark.parametrize("correction", [LongitudeLibrationCorrection.NONE, LongitudeLibrationCorrection.INPOP21A])
def test_analytic_lunar_angular_velocity_matches_the_attitude_derivative(correction):
    rates = np.array([2.1e-7, -3.2e-8, 2.4e-6])

    def angles(epoch):
        return np.r_[np.array([0.4, 0.7, -0.6]) + EPOCH.seconds_until(epoch) * rates, rates]

    orientation = EulerAngleLunarOrientation(
        angles, lambda epochs: np.array([angles(e) for e in epochs]), correction, EPOCH
    )
    epoch, h = EPOCH.shifted(123456.0), 4.0
    derivative = (
        orientation.pa_to_lcrs_matrix(epoch.shifted(h)) - orientation.pa_to_lcrs_matrix(epoch.shifted(-h))
    ) / (2 * h)
    skew = orientation.pa_to_lcrs_matrix(epoch).T @ derivative

    # Central differences with h = 4 s: truncation (h^2 |w|^3) and round off are below 3e-15 rad/s.
    assert_close(
        orientation.angular_velocity_rad_s(epoch),
        [skew[2, 1], skew[0, 2], skew[1, 0]],
        Tolerance("FINITE_DIFFERENCE_RAD_S", atol=3e-15, unit="rad/s"),
    )
