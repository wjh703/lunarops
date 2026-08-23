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


def test_epoch_keeps_two_part_jd_scale_and_supports_precise_shifts():
    epoch = Epoch(2451545.0, 0.25, TimeScale.TDB)
    shifted = epoch.shifted(2.5)

    assert (epoch.jd1, epoch.jd2) == (2451545.0, 0.25)
    assert shifted.scale is TimeScale.TDB
    assert epoch.seconds_until(shifted) == pytest.approx(2.5, abs=1.0e-10)
    assert Epoch(2458849.5, 0.0, TimeScale.UTC).date_iso() == "2020-01-01"


def test_tt_tdb_conversion_uses_erfa_and_round_trips():
    converter = TimeScaleConverter()
    expected_tdb = Epoch(2451545.0, 0.25, TimeScale.TDB)
    offset_s = float(erfa.dtdb(expected_tdb.jd1, expected_tdb.jd2, 0.0, 0.0, 0.0, 0.0))
    shifted = expected_tdb.shifted(-offset_s)
    tt = Epoch(shifted.jd1, shifted.jd2, TimeScale.TT)

    tdb = converter.tt2tdb(tt)
    recovered = converter.tdb2tt(tdb)

    assert expected_tdb.seconds_until(tdb) == pytest.approx(0.0, abs=1.0e-12)
    assert converter.tdb_minus_tt_s(tdb) == pytest.approx(
        float(erfa.dtdb(tdb.jd1, tdb.jd2, 0.0, 0.0, 0.0, 0.0)),
        abs=0.0,
    )
    assert tt.seconds_until(recovered) == pytest.approx(0.0, abs=1.0e-12)


def test_topocentric_tt_tdb_conversion_passes_erfa_station_arguments():
    converter = TimeScaleConverter()
    arguments = TdbTopocentricArguments(0.75, 0.4, 5100.0, 3800.0)
    expected_tdb = Epoch(2451545.0, 0.25, TimeScale.TDB)
    offset_s = float(
        erfa.dtdb(
            expected_tdb.jd1,
            expected_tdb.jd2,
            arguments.ut1_fraction_of_day,
            arguments.longitude_rad,
            arguments.distance_from_spin_axis_km,
            arguments.north_of_equatorial_plane_km,
        )
    )
    shifted = expected_tdb.shifted(-offset_s)
    epoch_tt = Epoch(shifted.jd1, shifted.jd2, TimeScale.TT)

    observer_epochs: list[Epoch] = []

    def topocentric_observer(epoch_utc: Epoch) -> TdbTopocentricArguments:
        observer_epochs.append(epoch_utc)
        return arguments

    recovered_tt = converter.tdb2tt(
        expected_tdb,
        topocentric_observer=topocentric_observer,
    )
    recovered_tdb = converter.tt2tdb(
        epoch_tt,
        topocentric_observer=lambda _epoch_utc: arguments,
    )

    assert len(observer_epochs) == 1
    assert observer_epochs[0].scale is TimeScale.UTC
    assert expected_tdb.seconds_until(recovered_tdb) == pytest.approx(0.0, abs=5.0e-12)
    assert epoch_tt.seconds_until(recovered_tt) == pytest.approx(0.0, abs=5.0e-12)


def test_epoch_rejects_implicit_scale_mixing():
    utc = Epoch(2451545.0, 0.0, TimeScale.UTC)
    tdb = Epoch(2451545.0, 0.0, TimeScale.TDB)

    with pytest.raises(ValueError, match="matching time scales"):
        utc.seconds_until(tdb)
    with pytest.raises(ValueError, match="TDB scale"):
        utc.require_scale(TimeScale.TDB)
    with pytest.raises(ValueError, match="ISOT output"):
        tdb.isot(TimeScaleConverter(), scale=TimeScale.TDB)
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


def test_fixed_utc_offset_normalizes_local_input_and_formats_local_output():
    utc = parse_time_with_utc_offset("2026-08-21T08:00:00", utc_offset_hours=8.0)
    assert utc is not None
    assert utc.isot(precision=0) == "2026-08-21T00:00:00"
    assert format_time_with_utc_offset(utc, utc_offset_hours=8.0, precision=0) == "2026-08-21T08:00:00+08:00"
    assert Epoch.from_isot("2026-08-21T08:00:00+08:00", scale=TimeScale.UTC).seconds_until(utc) == pytest.approx(
        0.0,
        abs=1.0e-9,
    )


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
    converter = TimeScaleConverter()
    utc = Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC)
    tt = converter.utc2tt(utc)
    tdb = converter.tt2tdb(tt)

    assert converter.tt2utc(tt).seconds_until(utc) == pytest.approx(0.0, abs=1.0e-9)
    assert tdb.isot(converter, scale=TimeScale.UTC).startswith("2020-01-01T00:00:00")
