from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from lunarops.classes.time import Epoch, TimeScale, TimeScaleConverter
from lunarops.classes.delays.shapiro import Iers2010ShapiroDelay
from lunarops.classes.ephemerides import (
    BodyState,
    CalcephEphemeris,
    Ephemeris,
    LongitudeLibrationCorrectionModel,
    LongitudeLibrationCorrectionType,
    LunarRelativisticScaleConvention,
    make_longitude_libration_correction_model,
    normalize_longitude_libration_correction_type,
    normalize_lunar_relativistic_scale_convention,
)
from lunarops.classes.frames import (
    EarthOrientationProvider,
    EarthOrientationSample,
    LunarFrameTransform,
    PolarMotion,
    ReferenceFrameSystem,
    RelativisticFrameTransform,
    TabulatedEarthOrientation,
)
from lunarops.classes.relativistic.constants import L_B_MINUS_L_L_LUNAR_SURFACE


class _FakeEphemeris(Ephemeris):
    def __init__(self):
        self.closed = False

    @property
    def source_file_path(self) -> Path:
        return Path("fake.eph")

    @property
    def l_b_minus_l_l(self) -> float:
        return 0.0

    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState:
        epoch_tdb.require_scale(TimeScale.TDB)
        index = {
            "SSB": 0,
            "SUN": 1,
            "EARTH": 2,
            "MOON": 3,
            "MERCURY BARYCENTER": 4,
            "VENUS BARYCENTER": 5,
            "MARS BARYCENTER": 6,
            "JUPITER BARYCENTER": 7,
            "SATURN BARYCENTER": 8,
            "URANUS BARYCENTER": 9,
            "NEPTUNE BARYCENTER": 10,
        }[body_name.upper()]
        velocity = np.array([12_000.0, -18_000.0, 3_000.0]) if index == 3 else np.zeros(3)
        return BodyState(
            position_m=np.array([index * 1.0e11, index * 1.0e9, 0.0]),
            velocity_mps=velocity,
        )

    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch_tdb.require_scale(TimeScale.TDB)
        return np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def geocentric_tdb_minus_tt_s(self, epoch_tdb: Epoch) -> float:
        epoch_tdb.require_scale(TimeScale.TDB)
        return 0.001

    def close(self) -> None:
        self.closed = True


class _FakeEarthOrientation(EarthOrientationProvider):
    def __init__(self):
        self.installed = False

    @property
    def source_file_path(self):
        return Path("fake.eop")

    def polar_motion(self, epoch_utc) -> PolarMotion:
        return PolarMotion(0.1, 0.2)

    def ut1_minus_utc_s(self, epoch_utc) -> float:
        return 0.0


def _tdb(jd2: float = 0.0) -> Epoch:
    return Epoch(2451545.0, jd2, TimeScale.TDB)


def test_epoch_and_body_state_are_frozen_and_validated():
    epoch = _tdb(0.25)
    state = BodyState(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

    assert (epoch.jd1, epoch.jd2) == (2451545.0, 0.25)
    assert epoch.scale is TimeScale.TDB
    assert not state.position_m.flags.writeable
    assert not state.velocity_mps.flags.writeable
    with pytest.raises(FrozenInstanceError):
        cast(Any, epoch).jd1 = 0.0
    with pytest.raises(ValueError):
        BodyState(np.array([1.0, 2.0]), np.zeros(3))


def test_lunar_frame_transform_round_trip_uses_ephemeris_orientation():
    transform = LunarFrameTransform(_FakeEphemeris())
    pa = np.array([1.0, 2.0, 3.0])

    lcrs = transform.pa2lcrs(pa, _tdb())
    recovered = transform.lcrs2pa(lcrs, _tdb())

    assert np.allclose(lcrs, [-2.0, 1.0, 3.0])
    assert np.allclose(recovered, pa)


def test_relativistic_frame_transform_round_trip_is_consistent():
    transform = RelativisticFrameTransform(_FakeEphemeris())
    gcrs = np.array([6_378_137.0, 100.0, -50.0])
    lcrs = np.array([1_737_400.0, -200.0, 75.0])

    bcrs = transform.gcrs2bcrs(gcrs, _tdb())
    recovered = transform.bcrs2gcrs(bcrs, _tdb())
    lunar_bcrs = transform.lcrs2bcrs(lcrs, _tdb())
    lunar_recovered = transform.bcrs2lcrs(lunar_bcrs, _tdb())

    assert np.all(np.isfinite(bcrs))
    assert np.allclose(recovered, gcrs, atol=1.0e-6)
    assert np.all(np.isfinite(lunar_bcrs))
    assert np.allclose(lunar_recovered, lcrs, atol=1.0e-6)


def test_reference_frame_system_owns_one_time_converter():
    ephemeris = _FakeEphemeris()
    earth_orientation = _FakeEarthOrientation()
    system = ReferenceFrameSystem(
        ephemeris=ephemeris,
        earth_orientation_provider=earth_orientation,
    )

    assert not earth_orientation.installed
    assert system.ephemeris.source_file_path == Path("fake.eph")
    assert isinstance(system.time_scale_converter, TimeScaleConverter)
    assert system.time_scale_converter.ephemeris is ephemeris
    assert np.allclose(system.pa2lcrs([1.0, 0.0, 0.0], _tdb()), [0.0, 1.0, 0.0])
    lunar_bcrs = system.lcrs2bcrs([1.0, 2.0, 3.0], _tdb())
    assert np.allclose(system.bcrs2lcrs(lunar_bcrs, _tdb()), [1.0, 2.0, 3.0])

    assert not ephemeris.closed


def test_zero_libration_factory_and_shapiro_use_epoch():
    epoch = _tdb()
    correction = make_longitude_libration_correction_model("none")
    assert isinstance(correction, LongitudeLibrationCorrectionModel)
    assert correction.correction_rad(epoch, j2000_epoch_tdb=epoch) == 0.0

    model = Iers2010ShapiroDelay(ephemeris=_FakeEphemeris())
    delay = model.path_delay_m(
        [2.0e11, 0.0, 0.0],
        [3.0e11, 0.0, 0.0],
        epoch,
    )
    assert np.isfinite(delay)
    assert delay >= 0.0


def test_longitude_libration_correction_type_normalization_is_explicit():
    assert normalize_longitude_libration_correction_type(None) is LongitudeLibrationCorrectionType.NONE
    assert normalize_longitude_libration_correction_type(" INPOP21A ") is LongitudeLibrationCorrectionType.INPOP21A
    for legacy_value in (False, True, "", "no", "off", "false", "0"):
        with pytest.raises((TypeError, ValueError)):
            normalize_longitude_libration_correction_type(cast(Any, legacy_value))


def test_ephemeris_exposes_libration_selection_as_enum():
    ephemeris = _FakeEphemeris()
    assert ephemeris.longitude_libration_correction_type is LongitudeLibrationCorrectionType.NONE


def test_lunar_relativistic_scale_convention_is_explicit_and_normalized():
    assert (
        normalize_lunar_relativistic_scale_convention(" tdbCompatibleLunarSurface ")
        is LunarRelativisticScaleConvention.TDB_COMPATIBLE_LUNAR_SURFACE
    )
    assert (
        normalize_lunar_relativistic_scale_convention("alreadyscaled")
        is LunarRelativisticScaleConvention.ALREADY_SCALED
    )
    with pytest.raises(ValueError, match="expected one of"):
        normalize_lunar_relativistic_scale_convention("inferredFromFilename")
    with pytest.raises(ValueError, match="expected one of"):
        normalize_lunar_relativistic_scale_convention("de440")


def test_calceph_lunar_scale_does_not_depend_on_filename(tmp_path, monkeypatch):
    class _FakeHandle:
        def close(self):
            return None

    class _FakeCalcephBin:
        @staticmethod
        def open(path):
            return _FakeHandle()

    class _FakeConstants:
        UNIT_KM = 1
        UNIT_SEC = 2
        USE_NAIFID = 4
        UNIT_RAD = 8

    monkeypatch.setitem(
        sys.modules,
        "calcephpy",
        SimpleNamespace(CalcephBin=_FakeCalcephBin, Constants=_FakeConstants),
    )
    de440_named_file = tmp_path / "de440.bsp"
    renamed_file = tmp_path / "renamed_kernel.bsp"
    de440_named_file.touch()
    renamed_file.touch()

    native_with_de440_name = CalcephEphemeris(
        de440_named_file,
        lunar_relativistic_scale_convention="alreadyScaled",
    )
    lunar_surface_scale_with_generic_name = CalcephEphemeris(
        renamed_file,
        lunar_relativistic_scale_convention="tdbCompatibleLunarSurface",
    )

    assert native_with_de440_name.l_b_minus_l_l == 0.0
    assert (
        lunar_surface_scale_with_generic_name.l_b_minus_l_l
        == L_B_MINUS_L_L_LUNAR_SURFACE
    )
    assert (
        native_with_de440_name.lunar_relativistic_scale_convention
        is LunarRelativisticScaleConvention.ALREADY_SCALED
    )
    assert (
        lunar_surface_scale_with_generic_name.lunar_relativistic_scale_convention
        is LunarRelativisticScaleConvention.TDB_COMPATIBLE_LUNAR_SURFACE
    )
    native_with_de440_name.close()
    lunar_surface_scale_with_generic_name.close()


def test_shapiro_validates_positions():
    model = Iers2010ShapiroDelay(ephemeris=_FakeEphemeris())
    with pytest.raises(ValueError, match="transmitter_bcrs_m"):
        model.path_delay_m([1.0, 2.0], [3.0, 4.0, 5.0], _tdb())


def test_c04_duplicate_mjd_policy_is_explicit():
    samples = (
        EarthOrientationSample(60000.0, 0.1, 0.2, 0.3),
        EarthOrientationSample(60000.0, 0.4, 0.5, 0.6),
        EarthOrientationSample(60001.0, 1.0, 1.1, 1.2),
    )
    with pytest.raises(ValueError, match="duplicateMjdPolicy"):
        TabulatedEarthOrientation(samples)

    eop = TabulatedEarthOrientation(samples, duplicate_mjd_policy="last")
    assert eop.duplicate_mjd_policy == "last"
    assert [sample.xp_arcsec for sample in eop.samples] == [0.4, 1.0]

    eop_mean = TabulatedEarthOrientation(samples, duplicate_mjd_policy="mean")
    assert [sample.xp_arcsec for sample in eop_mean.samples] == [0.25, 1.0]


def test_parse_eop_c04_and_finals_rows(tmp_path):
    from lunarops.classes.frames.earth_orientation import read_iers_eop

    path = tmp_path / "eop.txt"
    path.write_text(
        "# header\n"
        "1962 1 1 37665 0.123 0.456 0.789 0.0\n"
        "73 1 2 41684.00 I 0.120733 0.009786 0.136966 0.015902 I 0.8084176 0.0002710 3.5563 0.1916\n"
        "2020 1 1 0 58849 0.076 0.282 -0.177\n",
        encoding="utf-8",
    )
    samples = read_iers_eop(path)
    assert [sample.mjd_utc for sample in samples] == [37665.0, 41684.0, 58849.0]
    assert samples[0].xp_arcsec == 0.123
    assert samples[1].xp_arcsec == 0.120733
    assert samples[1].yp_arcsec == 0.136966
    assert samples[1].ut1_minus_utc_s == 0.8084176
    assert samples[2].ut1_minus_utc_s == -0.177


def test_eop_parse_error_includes_preview(tmp_path):
    from lunarops.classes.frames.earth_orientation import read_iers_eop

    path = tmp_path / "bad_eop.txt"
    path.write_text("not an eop row\nstill not eop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="First non-comment rows"):
        read_iers_eop(path)


def test_c04_mpi_payload_roundtrip_uses_arrays():
    samples = (
        EarthOrientationSample(60000.0, 0.1, 0.2, 0.3),
        EarthOrientationSample(60001.0, 0.4, 0.5, 0.6),
    )
    original = TabulatedEarthOrientation(samples, source_file_path="eop.txt")
    payload = original.to_mpi_payload()
    restored = TabulatedEarthOrientation.from_mpi_payload(payload)

    assert restored.source_file_path == original.source_file_path
    assert restored.mjd_utc_range == original.mjd_utc_range
    assert restored.samples == original.samples
    assert cast(Any, payload["mjdUtc"]).shape == (2,)


def test_terrestrial_transform_gcrs_itrf_round_trip(monkeypatch):
    from lunarops.classes.frames import TerrestrialFrameTransform

    transform = TerrestrialFrameTransform(_FakeEarthOrientation())
    matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    monkeypatch.setattr(
        TerrestrialFrameTransform,
        "gcrs2itrf_matrix",
        lambda self, epoch_utc: matrix,
    )
    gcrs = np.array([1.0, 2.0, 3.0])

    itrf = transform.gcrs2itrf(gcrs, Epoch(2451545.0, 0.0, TimeScale.UTC))
    recovered = transform.itrf2gcrs(itrf, Epoch(2451545.0, 0.0, TimeScale.UTC))

    assert np.allclose(itrf, [-2.0, 1.0, 3.0])
    assert np.allclose(recovered, gcrs)
