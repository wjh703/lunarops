from __future__ import annotations

import erfa
import numpy as np
import pytest

from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.time import Epoch, TimeScale
from scripts.compare_tdb_tt import compare_tdb_tt, erfa_tdb_minus_tt_s, tdb_epochs, tt2tdb_erfa


class _ErfaOffsetEphemeris(Ephemeris):
    @property
    def source_file_path(self):
        return None

    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState:
        return BodyState(np.zeros(3), np.zeros(3))

    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        return np.eye(3)

    def target16_tdb_minus_tt_s(self, epoch_tdb: Epoch) -> float:
        return erfa_tdb_minus_tt_s(epoch_tdb)


def test_erfa_tdb_minus_tt_is_evaluated_at_tdb_epoch():
    epoch_tdb = Epoch(2451545.0, 0.25, TimeScale.TDB)

    assert erfa_tdb_minus_tt_s(epoch_tdb) == pytest.approx(
        float(erfa.dtdb(epoch_tdb.jd1, epoch_tdb.jd2, 0.0, 0.0, 0.0, 0.0)),
        abs=0.0,
    )
    with pytest.raises(ValueError, match="TDB"):
        erfa_tdb_minus_tt_s(Epoch(2451545.0, 0.25, TimeScale.TT))


def test_tt_to_tdb_erfa_iterates_implicit_tdb_argument():
    expected_tdb = Epoch(2451545.0, 0.25, TimeScale.TDB)
    offset_s = erfa_tdb_minus_tt_s(expected_tdb)
    shifted = expected_tdb.shifted(-offset_s)
    epoch_tt = Epoch(shifted.jd1, shifted.jd2, TimeScale.TT)

    recovered_tdb = tt2tdb_erfa(epoch_tt)

    assert expected_tdb.seconds_until(recovered_tdb) == pytest.approx(0.0, abs=1.0e-12)


def test_comparison_uses_an_inclusive_tdb_grid_and_reports_tt_inverse():
    grid = list(tdb_epochs(2451545.0, 2451546.0, 0.6))
    rows = compare_tdb_tt(
        _ErfaOffsetEphemeris(),
        start_tdb_jd=2451545.0,
        end_tdb_jd=2451546.0,
        step_days=0.6,
    )

    assert [epoch.jd for epoch in grid] == pytest.approx([2451545.0, 2451545.6, 2451546.0])
    assert [row.tdb_jd for row in rows] == pytest.approx([epoch.jd for epoch in grid])
    for row in rows:
        assert row.ephemeris_minus_erfa_s == pytest.approx(0.0, abs=0.0)
        assert row.ephemeris_tt2tdb_minus_input_s == pytest.approx(0.0, abs=1.0e-12)
        assert row.erfa_tt2tdb_minus_input_s == pytest.approx(0.0, abs=1.0e-12)
