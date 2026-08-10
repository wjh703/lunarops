"""IERS 2010 solid-Earth tide station displacement."""

from __future__ import annotations

import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.frames import ReferenceFrameSystem

from .base import StationDisplacementInput


class Iers2010SolidEarthTide:
    """Solid-Earth tide displacement from the official IERS routine.

    ``DEHANTTIDEINEL`` requires geocentric ITRF vectors for the station, Sun,
    and Moon. The celestial vectors are derived from the configured ephemeris
    and the same frame system used by the observation model.
    """

    def __init__(self, frame_system: ReferenceFrameSystem) -> None:
        if not isinstance(frame_system, ReferenceFrameSystem):
            raise TypeError("frame_system must be a ReferenceFrameSystem.")
        self.frame_system = frame_system

    @staticmethod
    def _utc_calendar(epoch: Epoch) -> tuple[int, int, int, float]:
        epoch.require_scale(TimeScale.UTC, name="epoch_utc")
        date_text, time_text = epoch.isot(precision=9).split("T")
        year, month, day = (int(part) for part in date_text.split("-"))
        hour_text, minute_text, second_text = time_text.split(":")
        hour = int(hour_text)
        minute = int(minute_text)
        second = float(second_text)
        if second >= 60.0:
            raise ValueError("IERS DEHANTTIDEINEL cannot represent an exact UTC leap-second label.")
        fractional_hour = hour + minute / 60.0 + second / 3600.0
        return year, month, day, fractional_hour

    def _body_itrf_m(self, body: str, epoch_utc: Epoch, epoch_tdb: Epoch) -> np.ndarray:
        position_bcrs_m = self.frame_system.ephemeris.body_position_bcrs(body, epoch_tdb)
        position_gcrs_m = self.frame_system.bcrs2gcrs(position_bcrs_m, epoch_tdb)
        position_itrf_m = self.frame_system.gcrs2itrf(position_gcrs_m, epoch_utc)
        value = np.asarray(position_itrf_m, dtype=float).reshape(3)
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"Ephemeris/frame conversion returned non-finite {body} coordinates.")
        return value

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        epoch_utc = data.epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        epoch_tdb = self.frame_system.time_scale_converter.convert(epoch_utc, TimeScale.TDB)
        sun_itrf_m = self._body_itrf_m("SUN", epoch_utc, epoch_tdb)
        moon_itrf_m = self._body_itrf_m("MOON", epoch_utc, epoch_tdb)
        year, month, day, fractional_hour = self._utc_calendar(epoch_utc)

        displacement = np.asarray(
            _iers2010.dehanttideinel(
                data.reference_position_itrf_m,
                year,
                month,
                day,
                fractional_hour,
                sun_itrf_m,
                moon_itrf_m,
            ),
            dtype=float,
        )
        if displacement.size != 3 or not np.all(np.isfinite(displacement)):
            raise RuntimeError("DEHANTTIDEINEL returned an invalid displacement vector.")
        return displacement.reshape(3)
