"""Two-part Julian dates, UTC leap seconds and ERFA time-scale conversions."""

import erfa
import pytest

from lunarops.classes.time import (
    Epoch,
    TdbTopocentricArguments,
    TimeScale,
    TimeScaleConverter,
    format_time_with_utc_offset,
    parse_time_with_utc_offset,
    utc2tt,
)
from tests.tolerances import TIME_S, assert_close


def _tt_minus_utc_s(utc: Epoch) -> float:
    tt = utc2tt(utc)
    return ((tt.jd1 - utc.jd1) + (tt.jd2 - utc.jd2)) * 86400.0


def test_epoch_keeps_two_part_jd_and_scale_through_shifts():
    epoch = Epoch(2451545.0, 0.25, TimeScale.TDB)
    shifted = epoch.shifted(2.5)

    assert (epoch.jd1, epoch.jd2) == (2451545.0, 0.25)
    assert shifted.scale is TimeScale.TDB
    assert_close(epoch.seconds_until(shifted), 2.5, TIME_S)
    assert Epoch(2458849.5, 0.0, TimeScale.UTC).date_iso() == "2020-01-01"


def test_tt_tdb_conversion_matches_erfa_dtdb_and_round_trips():
    converter = TimeScaleConverter()
    expected_tdb = Epoch(2451545.0, 0.25, TimeScale.TDB)
    offset_s = float(erfa.dtdb(expected_tdb.jd1, expected_tdb.jd2, 0.0, 0.0, 0.0, 0.0))
    shifted = expected_tdb.shifted(-offset_s)
    tt = Epoch(shifted.jd1, shifted.jd2, TimeScale.TT)

    tdb = converter.tt2tdb(tt)

    assert_close(expected_tdb.seconds_until(tdb), 0.0, TIME_S)
    assert_close(converter.tdb_minus_tt_s(tdb), erfa.dtdb(tdb.jd1, tdb.jd2, 0.0, 0.0, 0.0, 0.0), TIME_S)
    assert_close(tt.seconds_until(converter.tdb2tt(tdb)), 0.0, TIME_S)


def test_topocentric_tdb_uses_station_arguments_at_utc():
    converter = TimeScaleConverter()
    arguments = TdbTopocentricArguments(0.75, 0.4, 5100.0, 3800.0)
    expected_tdb = Epoch(2451545.0, 0.25, TimeScale.TDB)
    offset_s = float(
        erfa.dtdb(
            expected_tdb.jd1,
            expected_tdb.jd2,
            *(
                arguments.ut1_fraction_of_day,
                arguments.longitude_rad,
                arguments.distance_from_spin_axis_km,
                arguments.north_of_equatorial_plane_km,
            ),
        )
    )
    shifted = expected_tdb.shifted(-offset_s)
    epoch_tt = Epoch(shifted.jd1, shifted.jd2, TimeScale.TT)
    observer_scales = []

    def observer(epoch_utc: Epoch) -> TdbTopocentricArguments:
        observer_scales.append(epoch_utc.scale)
        return arguments

    recovered_tt = converter.tdb2tt(expected_tdb, topocentric_observer=observer)
    recovered_tdb = converter.tt2tdb(epoch_tt, topocentric_observer=observer)

    assert set(observer_scales) == {TimeScale.UTC}
    assert_close(expected_tdb.seconds_until(recovered_tdb), 0.0, TIME_S)
    assert_close(epoch_tt.seconds_until(recovered_tt), 0.0, TIME_S)
    # The offsets are the ones the conversions apply, so recombining them reproduces the epochs exactly.
    from_tdb_s = converter.tdb_minus_tt_at_tdb_s(expected_tdb, topocentric_observer=observer)
    from_tt_s = converter.tdb_minus_tt_at_tt_s(epoch_tt, topocentric_observer=observer)
    assert from_tdb_s == offset_s
    assert_close(from_tt_s, offset_s, TIME_S)
    assert recovered_tt == Epoch(*_parts(expected_tdb.shifted(-from_tdb_s)), TimeScale.TT)
    assert recovered_tdb == Epoch(*_parts(epoch_tt.shifted(from_tt_s)), TimeScale.TDB)


def _parts(epoch: Epoch) -> tuple[float, float]:
    return epoch.jd1, epoch.jd2


@pytest.mark.parametrize(
    ("isot", "tt_minus_utc_s"),
    [
        # TT-TAI = 32.184 s; TAI-UTC from the IERS Bulletin C table.
        ("1961-01-01T00:00:00", 32.184 + 1.422818),  # pre-1972 drift segment
        ("1999-01-01T00:00:00", 32.184 + 32.0),
        ("2017-06-01T00:00:00", 32.184 + 37.0),
    ],
)
def test_utc_to_tt_follows_published_tai_minus_utc(isot, tt_minus_utc_s):
    assert_close(_tt_minus_utc_s(Epoch.from_isot(isot, scale=TimeScale.UTC)), tt_minus_utc_s, TIME_S)


def test_utc_elapsed_seconds_count_the_leap_second():
    before = Epoch.from_isot("2016-12-31T23:59:59", scale=TimeScale.UTC)
    leap = Epoch.from_isot("2016-12-31T23:59:60", scale=TimeScale.UTC)
    after = Epoch.from_isot("2017-01-01T00:00:00", scale=TimeScale.UTC)

    assert leap.isot(scale=TimeScale.UTC, precision=3) == "2016-12-31T23:59:60.000"
    assert Epoch.from_date_seconds("20161231", 86400.0).isot(precision=3) == "2016-12-31T23:59:60.000"
    assert_close(before.seconds_until(leap), 1.0, TIME_S)
    assert_close(leap.seconds_until(after), 1.0, TIME_S)
    assert_close(before.seconds_until(after), 2.0, TIME_S)
    assert before.shifted(2.0).isot(precision=3) == "2017-01-01T00:00:00.000"


def test_civil_constructors_agree():
    from_calendar = Epoch.from_calendar(2020, 1, 2, 3, 4, 5.25)
    from_date_seconds = Epoch.from_date_seconds("20200102", 3 * 3600 + 4 * 60 + 5.25)
    from_isot = Epoch.from_isot("2020-01-02T03:04:05.25", scale=TimeScale.UTC)

    assert from_calendar.scale is TimeScale.UTC
    assert_close(from_calendar.seconds_until(from_date_seconds), 0.0, TIME_S)
    assert_close(from_calendar.seconds_until(from_isot), 0.0, TIME_S)
    assert from_calendar.isot(precision=3) == "2020-01-02T03:04:05.250"


def test_fixed_utc_offset_parses_local_time_and_formats_it_back():
    utc = parse_time_with_utc_offset("2026-08-21T08:00:00", utc_offset_hours=8.0)

    assert utc.isot(precision=0) == "2026-08-21T00:00:00"
    assert format_time_with_utc_offset(utc, utc_offset_hours=8.0, precision=0) == "2026-08-21T08:00:00+08:00"
    assert_close(Epoch.from_isot("2026-08-21T08:00:00+08:00", scale=TimeScale.UTC).seconds_until(utc), 0.0, TIME_S)


def test_converter_routes_utc_tt_and_tdb():
    converter = TimeScaleConverter()
    utc = Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC)
    tt = converter.utc2tt(utc)
    tdb = converter.tt2tdb(tt)

    assert_close(converter.tt2utc(tt).seconds_until(utc), 0.0, TIME_S)
    assert converter.convert(tdb, TimeScale.UTC).isot().startswith("2020-01-01T00:00:00")


def test_time_scales_are_never_mixed_implicitly():
    utc = Epoch(2451545.0, 0.0, TimeScale.UTC)
    tdb = Epoch(2451545.0, 0.0, TimeScale.TDB)

    with pytest.raises(ValueError, match="matching time scales"):
        utc.seconds_until(tdb)
    with pytest.raises(ValueError, match="TDB scale"):
        utc.require_scale(TimeScale.TDB)
    with pytest.raises(ValueError, match="comparisons require matching time scales"):
        _ = utc == tdb
    with pytest.raises(ValueError, match="ISOT output"):
        tdb.isot(scale=TimeScale.TDB)
    with pytest.raises(ValueError, match="TimeScaleConverter"):
        tdb.isot()
    with pytest.raises(ValueError, match="must use the UTC scale"):
        tdb.date_iso()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: Epoch.from_isot("2020-01-01", scale="tdb"),
        lambda: Epoch.from_calendar(2020, 1, 1, scale="tdb"),
        lambda: Epoch.from_date_seconds("20200101", 0.0, scale="tdb"),
    ],
)
def test_tdb_has_no_civil_constructor(constructor):
    with pytest.raises(ValueError, match="TDB has no civil constructor"):
        constructor()


@pytest.mark.parametrize("second", [-0.1, 60.0, 60.5, 61.0, float("nan"), float("inf")])
def test_calendar_rejects_invalid_seconds_on_a_non_leap_day(second):
    with pytest.raises(ValueError):
        Epoch.from_calendar(2020, 1, 1, 23, 59, second)


def test_erfa_rejects_invalid_leap_labels_and_dubious_years():
    with pytest.raises(ValueError, match="time is after end of day"):
        Epoch.from_isot("2016-12-30T23:59:60", scale=TimeScale.UTC)
    with pytest.raises(ValueError, match="time is after end of day"):
        Epoch.from_date_seconds("20161230", 86400.0, scale=TimeScale.UTC)
    with pytest.raises(ValueError, match="dubious year"):
        Epoch.from_isot("2500-01-01T00:00:00", scale=TimeScale.UTC)
