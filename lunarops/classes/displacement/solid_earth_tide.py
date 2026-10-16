"""IERS 2010 solid-Earth tide station displacement."""

from __future__ import annotations

import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.time import Epoch, TimeScale

from .base import StationDisplacementInput


class Iers2010SolidEarthTide:
    """Solid-Earth tide displacement from the official IERS routine.

    ``DEHANTTIDEINEL`` requires geocentric ITRF vectors for the station, Sun,
    and Moon. The celestial vectors are derived from the configured ephemeris
    and the same frame system used by the observation model.
    """

    def __init__(self, frame_system: ReferenceFrameSystem) -> None:
        self.frame_system = frame_system

    @staticmethod
    def _utc_calendar(epoch: Epoch) -> tuple[int, int, int, float]:
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

    def _body_itrf_m(self, body: str, epoch_tdb: Epoch, matrix) -> np.ndarray:
        position_bcrs_m = self.frame_system.ephemeris.body_position_bcrs(body, epoch_tdb)
        position_gcrs_m = self.frame_system.relativistic_transform.bcrs2gcrs(position_bcrs_m, epoch_tdb)
        return matrix @ position_gcrs_m

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        epoch_utc = data.epoch_utc
        epoch_tdb = self.frame_system.time_scale_converter.convert(epoch_utc, TimeScale.TDB)
        matrix = self.frame_system.terrestrial_transform.gcrs2itrf_matrix(epoch_utc)
        sun_itrf_m = self._body_itrf_m("SUN", epoch_tdb, matrix)
        moon_itrf_m = self._body_itrf_m("MOON", epoch_tdb, matrix)
        year, month, day, fractional_hour = self._utc_calendar(epoch_utc)

        return np.asarray(
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
