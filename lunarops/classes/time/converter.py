"""Explicit UTC/TT/TDB conversion services backed by ERFA.

UTC<->TT is handled by the ERFA-backed routines in :mod:`.epoch`. TT<->TDB
uses ERFA ``dtdb``. Its formal independent variable is TDB, so TT->TDB is
solved by fixed-point iteration. The optional topocentric arguments are scalar
data supplied by the frames/observation layer, avoiding a dependency from this
low-level module back to Earth-orientation services.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import erfa

from .epoch import Epoch, TimeScale
from .epoch import tt2utc as _tt2utc
from .epoch import utc2tt as _utc2tt

_MAX_TDB_ITERATIONS = 6
_TDB_TT_TOLERANCE_S = 1.0e-12


@dataclass(frozen=True, slots=True)
class TdbTopocentricArguments:
    """Station arguments required by ERFA ``dtdb`` at one UTC epoch."""

    ut1_fraction_of_day: float
    longitude_rad: float
    distance_from_spin_axis_km: float
    north_of_equatorial_plane_km: float

    def __post_init__(self) -> None:
        for name in (
            "ut1_fraction_of_day",
            "longitude_rad",
            "distance_from_spin_axis_km",
            "north_of_equatorial_plane_km",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.distance_from_spin_axis_km < 0.0:
            raise ValueError("distance_from_spin_axis_km must not be negative.")
        object.__setattr__(self, "ut1_fraction_of_day", self.ut1_fraction_of_day % 1.0)


TdbTopocentricArgumentsProvider = Callable[[Epoch], TdbTopocentricArguments]


class TimeScaleConverter:
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
        topocentric_arguments: TdbTopocentricArguments | None = None,
    ) -> float:
        """Return ERFA TDB-TT at TDB, optionally including a station term."""

        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        if topocentric_arguments is None:
            ut1_fraction_of_day = longitude_rad = u_km = v_km = 0.0
        else:
            arguments = _require_topocentric_arguments(topocentric_arguments)
            ut1_fraction_of_day = arguments.ut1_fraction_of_day
            longitude_rad = arguments.longitude_rad
            u_km = arguments.distance_from_spin_axis_km
            v_km = arguments.north_of_equatorial_plane_km
        return float(
            erfa.dtdb(
                epoch_tdb.jd1,
                epoch_tdb.jd2,
                ut1_fraction_of_day,
                longitude_rad,
                u_km,
                v_km,
            )
        )

    def tdb2tt(
        self,
        epoch_tdb: Epoch,
        *,
        topocentric_observer: TdbTopocentricArgumentsProvider | None = None,
    ) -> Epoch:
        """Convert TDB to TT directly after sampling a UTC-dependent observer."""

        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        if topocentric_observer is None:
            return _tdb_minus_offset(epoch_tdb, self.tdb_minus_tt_s(epoch_tdb))

        # The dtdb date argument is TDB.  This preliminary geocentric result
        # only supplies UTC for C04/EOP and station-coordinate sampling.
        preliminary_tt = _tdb_minus_offset(epoch_tdb, self.tdb_minus_tt_s(epoch_tdb))
        arguments = _topocentric_arguments_at_utc(
            topocentric_observer,
            self.tt2utc(preliminary_tt),
        )
        return _tdb_minus_offset(
            epoch_tdb,
            self.tdb_minus_tt_s(epoch_tdb, topocentric_arguments=arguments),
        )

    def tt2tdb(
        self,
        epoch_tt: Epoch,
        *,
        topocentric_observer: TdbTopocentricArgumentsProvider | None = None,
    ) -> Epoch:
        """Convert TT to TDB by iterating ERFA's TDB-dependent relation."""

        epoch_tt.require_scale(TimeScale.TT, name="epoch_tt")
        arguments = (
            None
            if topocentric_observer is None
            else _topocentric_arguments_at_utc(topocentric_observer, self.tt2utc(epoch_tt))
        )
        current = Epoch(epoch_tt.jd1, epoch_tt.jd2, TimeScale.TDB)
        for _ in range(_MAX_TDB_ITERATIONS):
            delta_s = self.tdb_minus_tt_s(current, topocentric_arguments=arguments)
            updated = _tt_plus_offset(epoch_tt, delta_s)
            if abs(current.seconds_until(updated)) < _TDB_TT_TOLERANCE_S:
                return updated
            current = updated
        return current

    def convert(
        self,
        epoch: Epoch,
        scale: TimeScale | str,
        *,
        topocentric_observer: TdbTopocentricArgumentsProvider | None = None,
    ) -> Epoch:
        target = TimeScale.parse(scale)
        if epoch.scale is target:
            return epoch
        if epoch.scale is TimeScale.UTC and target is TimeScale.TT:
            return self.utc2tt(epoch)
        if epoch.scale is TimeScale.TT and target is TimeScale.UTC:
            return self.tt2utc(epoch)
        if epoch.scale is TimeScale.TT and target is TimeScale.TDB:
            return self.tt2tdb(epoch, topocentric_observer=topocentric_observer)
        if epoch.scale is TimeScale.TDB and target is TimeScale.TT:
            return self.tdb2tt(epoch, topocentric_observer=topocentric_observer)
        if epoch.scale is TimeScale.UTC and target is TimeScale.TDB:
            return self.tt2tdb(
                self.utc2tt(epoch),
                topocentric_observer=topocentric_observer,
            )
        if epoch.scale is TimeScale.TDB and target is TimeScale.UTC:
            return self.tt2utc(self.tdb2tt(epoch, topocentric_observer=topocentric_observer))
        raise AssertionError("Unhandled time-scale conversion.")


def _require_topocentric_arguments(value: TdbTopocentricArguments) -> TdbTopocentricArguments:
    if not isinstance(value, TdbTopocentricArguments):
        raise TypeError("topocentric arguments must be TdbTopocentricArguments.")
    return value


def _topocentric_arguments_at_utc(
    provider: TdbTopocentricArgumentsProvider,
    epoch_utc: Epoch,
) -> TdbTopocentricArguments:
    epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
    return _require_topocentric_arguments(provider(epoch_utc))


def _tdb_minus_offset(epoch_tdb: Epoch, offset_s: float) -> Epoch:
    shifted = epoch_tdb.shifted(-offset_s)
    return Epoch(shifted.jd1, shifted.jd2, TimeScale.TT)


def _tt_plus_offset(epoch_tt: Epoch, offset_s: float) -> Epoch:
    shifted = epoch_tt.shifted(offset_s)
    return Epoch(shifted.jd1, shifted.jd2, TimeScale.TDB)


__all__ = ["TdbTopocentricArguments", "TdbTopocentricArgumentsProvider", "TimeScaleConverter"]
