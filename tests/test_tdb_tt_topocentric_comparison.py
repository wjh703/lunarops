from __future__ import annotations

import math

import erfa
import numpy as np
import pytest

from lunarops.classes.time import Epoch, TimeScale
from scripts.compare_tdb_tt_topocentric import (
    erfa_topocentric_tdb_minus_tt_s,
    site_parameters_from_itrf,
    ut1_fraction_of_day,
)


def test_erfa_topocentric_term_excludes_geocentric_dtdb_component():
    epoch_tdb = Epoch(2451545.0, 0.25, TimeScale.TDB)
    ut1_fraction = 0.75
    elong_rad, u_km, v_km = 0.4, 5100.0, 3800.0

    actual = erfa_topocentric_tdb_minus_tt_s(
        epoch_tdb,
        ut1_fraction_of_day=ut1_fraction,
        elong_rad=elong_rad,
        u_km=u_km,
        v_km=v_km,
    )
    expected = float(
        erfa.dtdb(epoch_tdb.jd1, epoch_tdb.jd2, ut1_fraction, elong_rad, u_km, v_km)
        - erfa.dtdb(epoch_tdb.jd1, epoch_tdb.jd2, 0.0, 0.0, 0.0, 0.0)
    )

    assert actual == pytest.approx(expected, abs=0.0)


def test_site_parameters_use_itrf_geocentric_coordinates():
    elong_rad, u_km, v_km = site_parameters_from_itrf(np.array([3_000_000.0, 4_000_000.0, 5_000_000.0]))

    assert elong_rad == pytest.approx(math.atan2(4.0, 3.0))
    assert u_km == pytest.approx(5000.0)
    assert v_km == pytest.approx(5000.0)


def test_ut1_fraction_uses_midnight_not_julian_date_noon():
    class _EarthOrientation:
        def ut1_minus_utc_s(self, epoch_utc: Epoch) -> float:
            return 0.0

    epoch_utc = Epoch(2451545.0, 0.25, TimeScale.UTC)
    # Disable the IERS high-frequency evaluation in this unit-level test.
    expected = (epoch_utc.jd1 % 1.0 + epoch_utc.jd2 % 1.0 + 0.5) % 1.0

    # The concrete EOP type is only used for its UT1 method at runtime.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "scripts.compare_tdb_tt_topocentric.high_frequency_eop_correction",
            lambda epoch_utc, background_ut1_minus_utc_s: type("Correction", (), {"delta_ut1_s": 0.0})(),
        )
        assert ut1_fraction_of_day(epoch_utc, _EarthOrientation()) == pytest.approx(expected)
