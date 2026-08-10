"""Explicit UTC/TT/TDB conversion services.

UTC<->TT is handled by the ERFA-backed routines in :mod:`.epoch`. TT<->TDB
depends on the configured ephemeris target-16 table and, optionally, the
topocentric ``v_E dot X / c^2`` term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.array_validation import vector3
from lunarops.base.constants import C2
from .epoch import Epoch, TimeScale
from .epoch import tt2utc as _tt2utc
from .epoch import utc2tt as _utc2tt

if TYPE_CHECKING:
    from lunarops.classes.ephemerides.base import Ephemeris

_MAX_TDB_ITERATIONS = 6
_TDB_TT_TOLERANCE_S = 1.0e-12


class TimeScaleConverter:
    def __init__(self, ephemeris: Ephemeris) -> None:
        self.ephemeris = ephemeris

    def utc2tt(self, epoch: Epoch) -> Epoch:
        epoch.require_scale(TimeScale.UTC)
        return _utc2tt(epoch)

    def tt2utc(self, epoch: Epoch) -> Epoch:
        epoch.require_scale(TimeScale.TT)
        return _tt2utc(epoch)

    def tdb_minus_tt_s(
        self,
        epoch_tdb: Epoch,
        *,
        station_gcrs_m: ArrayLike | None = None,
    ) -> float:
        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        ephemeris = self.ephemeris
        geocentric = ephemeris.geocentric_tdb_minus_tt_s(epoch_tdb)
        if geocentric is None:
            raise RuntimeError("The configured ephemeris does not provide a TDB-TT table.")
        correction = 0.0
        if station_gcrs_m is not None:
            station = vector3(station_gcrs_m, name="station_gcrs_m")
            earth_velocity = ephemeris.body_state_bcrs(
                "EARTH",
                epoch_tdb,
            ).velocity_mps
            correction = float(np.dot(earth_velocity, station)) / C2
        return float(geocentric) + correction

    def tdb2tt(
        self,
        epoch_tdb: Epoch,
        *,
        station_gcrs_m: ArrayLike | None = None,
    ) -> Epoch:
        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        delta_s = self.tdb_minus_tt_s(
            epoch_tdb,
            station_gcrs_m=station_gcrs_m,
        )
        shifted = epoch_tdb.shifted(-delta_s)
        return Epoch(shifted.jd1, shifted.jd2, TimeScale.TT)

    def tt2tdb(
        self,
        epoch_tt: Epoch,
        *,
        station_gcrs_m: ArrayLike | None = None,
    ) -> Epoch:
        epoch_tt.require_scale(TimeScale.TT, name="epoch_tt")
        current = Epoch(epoch_tt.jd1, epoch_tt.jd2, TimeScale.TDB)
        for _ in range(_MAX_TDB_ITERATIONS):
            delta_s = self.tdb_minus_tt_s(
                current,
                station_gcrs_m=station_gcrs_m,
            )
            shifted = epoch_tt.shifted(delta_s)
            updated = Epoch(shifted.jd1, shifted.jd2, TimeScale.TDB)
            if abs(current.seconds_until(updated)) < _TDB_TT_TOLERANCE_S:
                return updated
            current = updated
        return current

    def convert(
        self,
        epoch: Epoch,
        scale: TimeScale | str,
        *,
        station_gcrs_m: ArrayLike | None = None,
    ) -> Epoch:
        target = TimeScale.parse(scale)
        if epoch.scale is target:
            return epoch
        if epoch.scale is TimeScale.UTC and target is TimeScale.TT:
            return self.utc2tt(epoch)
        if epoch.scale is TimeScale.TT and target is TimeScale.UTC:
            return self.tt2utc(epoch)
        if epoch.scale is TimeScale.TT and target is TimeScale.TDB:
            return self.tt2tdb(epoch, station_gcrs_m=station_gcrs_m)
        if epoch.scale is TimeScale.TDB and target is TimeScale.TT:
            return self.tdb2tt(epoch, station_gcrs_m=station_gcrs_m)
        if epoch.scale is TimeScale.UTC and target is TimeScale.TDB:
            return self.tt2tdb(
                self.utc2tt(epoch),
                station_gcrs_m=station_gcrs_m,
            )
        if epoch.scale is TimeScale.TDB and target is TimeScale.UTC:
            return self.tt2utc(self.tdb2tt(epoch, station_gcrs_m=station_gcrs_m))
        raise AssertionError("Unhandled time-scale conversion.")


__all__ = ["TimeScaleConverter"]
