import numpy as np
import pytest

from lunarops.classes.time import Epoch, TimeScale, TimeScaleConverter, utc2tt
from lunarops.classes.ephemerides import BodyState, Ephemeris


class _ConstantOffsetEphemeris(Ephemeris):
    @property
    def source_file_path(self):
        return None

    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState:
        return BodyState(np.zeros(3), np.zeros(3))

    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        return np.eye(3)

    def geocentric_tdb_minus_tt_s(self, epoch_tdb: Epoch) -> float:
        epoch_tdb.require_scale(TimeScale.TDB)
        return 0.0015


def test_epoch_keeps_two_part_jd_scale_and_supports_precise_shifts():
    epoch = Epoch(2451545.0, 0.25, TimeScale.TDB)
    shifted = epoch.shifted(2.5)

    assert (epoch.jd1, epoch.jd2) == (2451545.0, 0.25)
    assert shifted.scale is TimeScale.TDB
    assert epoch.seconds_until(shifted) == pytest.approx(2.5, abs=1.0e-10)
    assert Epoch(2458849.5, 0.0, TimeScale.UTC).date_iso() == "2020-01-01"


def test_tt_tdb_conversion_uses_ephemeris_table_and_round_trips():
    ephemeris = _ConstantOffsetEphemeris()
    converter = TimeScaleConverter(ephemeris)
    tt = Epoch(2451545.0, 0.0, TimeScale.TT)

    tdb = converter.tt2tdb(tt)
    recovered = converter.tdb2tt(tdb)

    assert tt.seconds_until(Epoch(tdb.jd1, tdb.jd2, TimeScale.TT)) == pytest.approx(0.0015, abs=1.0e-10)
    assert tt.seconds_until(recovered) == pytest.approx(0.0, abs=1.0e-10)
    assert ephemeris.source_file_path is None


def test_epoch_rejects_implicit_scale_mixing():
    utc = Epoch(2451545.0, 0.0, TimeScale.UTC)
    tdb = Epoch(2451545.0, 0.0, TimeScale.TDB)

    with pytest.raises(ValueError, match="matching time scales"):
        utc.seconds_until(tdb)
    with pytest.raises(ValueError, match="TDB scale"):
        utc.require_scale(TimeScale.TDB)
    with pytest.raises(ValueError, match="ISOT output"):
        tdb.isot(TimeScaleConverter(_ConstantOffsetEphemeris()), scale=TimeScale.TDB)
    with pytest.raises(ValueError, match="comparisons require matching time scales"):
        _ = utc == tdb


def test_tdb_civil_construction_and_direct_foreign_time_export_are_forbidden():
    tdb = Epoch(2451545.0, 0.0, TimeScale.TDB)

    with pytest.raises(ValueError, match="no direct civil/ISOT constructor"):
        Epoch.from_isot("2000-01-01T12:00:00", scale=TimeScale.TDB)
    assert not hasattr(tdb, "to_astropy")
    with pytest.raises(ValueError, match="must use the UTC scale"):
        tdb.date_iso()


def test_file_input_classmethods_without_astropy_dependency():
    from_calendar = Epoch.from_calendar(2020, 1, 2, 3, 4, 5.25)
    from_date_seconds = Epoch.from_date_seconds("20200102", 3 * 3600 + 4 * 60 + 5.25)

    assert from_calendar.scale is TimeScale.UTC
    assert from_calendar.seconds_until(from_date_seconds) == pytest.approx(0.0, abs=1.0e-9)
    assert from_calendar.isot(scale=TimeScale.UTC).startswith("2020-01-02T03:04:05.250")


def test_utc_elapsed_seconds_respect_leap_seconds_without_astropy_dependency():
    before = Epoch.from_isot("2016-12-31T23:59:59", scale=TimeScale.UTC)
    after = before.shifted(2.0)

    assert before.seconds_until(after) == pytest.approx(2.0, abs=1.0e-12)
    assert after.isot(scale=TimeScale.UTC).startswith("2017-01-01T00:00:00")


def test_utc_leap_second_label_round_trips_through_erfa():
    before = Epoch.from_isot("2016-12-31T23:59:59", scale=TimeScale.UTC)
    leap = Epoch.from_isot("2016-12-31T23:59:60", scale=TimeScale.UTC)
    leap_from_seconds = Epoch.from_date_seconds(
        "20161231",
        86400.0,
        scale=TimeScale.UTC,
    )
    after = Epoch.from_isot("2017-01-01T00:00:00", scale=TimeScale.UTC)

    assert leap.isot(scale=TimeScale.UTC, precision=3) == "2016-12-31T23:59:60.000"
    assert leap_from_seconds.isot(scale=TimeScale.UTC, precision=3) == ("2016-12-31T23:59:60.000")
    assert before.seconds_until(leap) == pytest.approx(1.0, abs=1.0e-11)
    assert leap.seconds_until(after) == pytest.approx(1.0, abs=1.0e-11)
    assert leap.shifted(1.0).seconds_until(after) == pytest.approx(0.0, abs=1.0e-11)


def test_erfa_utc_model_covers_pre_1972_drift():
    utc = Epoch.from_isot("1961-01-01T00:00:00", scale=TimeScale.UTC)
    tt = utc2tt(utc)
    tt_minus_utc_s = ((tt.jd1 - utc.jd1) + (tt.jd2 - utc.jd2)) * 86400.0

    assert tt_minus_utc_s == pytest.approx(32.184 + 1.422818, abs=1.0e-12)


def test_erfa_rejects_invalid_leap_labels_and_dubious_utc_years():
    with pytest.raises(ValueError, match="time is after end of day"):
        Epoch.from_isot("2016-12-30T23:59:60", scale=TimeScale.UTC)
    with pytest.raises(ValueError, match="time is after end of day"):
        Epoch.from_date_seconds("20161230", 86400.0, scale=TimeScale.UTC)
    with pytest.raises(ValueError, match="dubious year"):
        Epoch.from_isot("2500-01-01T00:00:00", scale=TimeScale.UTC)


def test_utc_tt_and_tdb_isot_route_through_converter_without_astropy_dependency():
    converter = TimeScaleConverter(_ConstantOffsetEphemeris())
    utc = Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC)
    tt = converter.utc2tt(utc)
    tdb = converter.tt2tdb(tt)

    assert converter.tt2utc(tt).seconds_until(utc) == pytest.approx(0.0, abs=1.0e-9)
    assert tdb.isot(converter, scale=TimeScale.UTC).startswith("2020-01-01T00:00:00")
