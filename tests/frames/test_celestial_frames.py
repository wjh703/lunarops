"""Body identifiers, lunar orientation, BCRS/GCRS/LCRS transforms and the CALCEPH adapter."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lunarops.classes.ephemerides import (
    BodyId,
    CalcephEphemeris,
    FixedLunarOrientation,
    LunarRelativisticScale,
    canonical_body_id,
    naif_id,
)
from lunarops.classes.ephemerides.body_ids import BODY_BY_NAIF_ID, NAIF_ID_BY_BODY
from lunarops.classes.ephemerides.lunar_orientation import (
    CalcephLunarOrientation,
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

    orientation = CalcephLunarOrientation(
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


class _FakeCalceph:
    """CALCEPH stand-in: positions in km equal (target, 2*target, 3*target), velocities a tenth of that."""

    UNIT_KM, UNIT_SEC, USE_NAIFID, UNIT_RAD, TDB = 1, 2, 4, 8, 64

    def __init__(self, opened):
        self.opened = opened

    def open(self, paths):
        self.opened.extend(Path(path).suffix for path in paths)
        return self

    def prefetch(self):
        return None

    def gettimescale(self):
        return self.TDB

    def getorientrecordcount(self):
        return 1

    def getorientrecordindex2(self, index):
        return (31008, 2400000.5, 2500000.5, 1, 2)

    def getpositionrecordcount(self):
        return 2

    def getpositionrecordindex2(self, index):
        return {1: (4, 0, 2400000.5, 2500000.5, 1, 2), 2: (10, 0, 2400000.5, 2500000.5, 1, 2)}[index]

    def compute_unit(self, jd1, jd2, target, center, units):
        state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]) * target
        return np.repeat(state[:, None], len(jd1), axis=1) if np.ndim(jd1) else state

    def compute_order(self, jd1, jd2, target, center, units, order):
        return np.vstack([np.zeros((6, len(jd1))), np.full((3, len(jd1)), float(target))])

    def orient_unit(self, jd1, jd2, target, units):
        return np.zeros((6, len(jd1)))

    def close(self):
        return None


def test_calceph_adapter_opens_position_and_orientation_kernels_and_converts_to_si(tmp_path, monkeypatch):
    opened: list[str] = []
    fake = _FakeCalceph(opened)
    monkeypatch.setitem(sys.modules, "calcephpy", SimpleNamespace(CalcephBin=fake, Constants=fake))
    for name in ("positions.bsp", "attitude.bpc", "frames.tf", "constants.tpc"):
        (tmp_path / name).touch()

    ephemeris = CalcephEphemeris(tmp_path, lunar_relativistic_scale_convention="alreadyScaled")
    state = ephemeris.body_state_bcrs("SUN", EPOCH)
    accelerations = ephemeris.body_accelerations_bcrs("SUN", (EPOCH, EPOCH))

    assert opened == [".bsp", ".bpc"]
    assert_close(state.position_m, [10_000.0, 20_000.0, 30_000.0], REL)
    assert_close(state.velocity_mps, [1_000.0, 2_000.0, 3_000.0], REL)
    assert_close(accelerations, np.full((2, 3), 10_000.0), REL)
    assert accelerations.flags.c_contiguous  # consumed directly by the native RHS
    assert_close(ephemeris.lunar_orientation.pa_to_lcrs_matrix(EPOCH), np.eye(3), REL)
    with pytest.raises(KeyError, match="try MARS BARYCENTER"):
        ephemeris.body_state_bcrs("MARS", EPOCH)
    ephemeris.close()


def test_calceph_adapter_requires_a_directory_with_bsp_and_bpc_kernels(tmp_path):
    (tmp_path / "positions.bsp").touch()
    with pytest.raises(ValueError, match="must contain .bsp and .bpc"):
        CalcephEphemeris(tmp_path, lunar_relativistic_scale_convention="alreadyScaled")
    with pytest.raises(ValueError, match="kernel directory not found"):
        CalcephEphemeris(tmp_path / "positions.bsp", lunar_relativistic_scale_convention="alreadyScaled")
