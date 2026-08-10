"""Optional longitude-libration corrections applied to lunar orientation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.time import Epoch

from .base import LongitudeLibrationCorrectionType, require_tdb_epoch

_MAS2RAD = np.deg2rad(1.0 / 3_600_000.0)
_JULIAN_CENTURY_DAYS = 36525.0


def normalize_longitude_libration_correction_type(
    value: LongitudeLibrationCorrectionType | str | None,
) -> LongitudeLibrationCorrectionType:
    if isinstance(value, LongitudeLibrationCorrectionType):
        return value
    if value is None:
        return LongitudeLibrationCorrectionType.NONE
    if not isinstance(value, str):
        raise TypeError(
            "longitude-libration correction type must be a string, LongitudeLibrationCorrectionType, or None."
        )
    text = value.strip().lower()
    try:
        return LongitudeLibrationCorrectionType(text)
    except ValueError:
        allowed = ", ".join(correction_type.value for correction_type in LongitudeLibrationCorrectionType)
        raise ValueError(
            f"Unsupported longitude-libration correction type {value!r}; expected one of: {allowed}."
        ) from None


@runtime_checkable
class LongitudeLibrationCorrectionModel(Protocol):
    def correction_rad(
        self,
        epoch_tdb: Epoch,
        *,
        j2000_epoch_tdb: Epoch,
    ) -> float: ...


class ZeroLongitudeLibrationCorrection:
    def correction_rad(
        self,
        epoch_tdb: Epoch,
        *,
        j2000_epoch_tdb: Epoch,
    ) -> float:
        require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        require_tdb_epoch(j2000_epoch_tdb, name="j2000_epoch_tdb")
        return 0.0


class Inpop21aLongitudeLibrationCorrection:
    """INPOP21a Table 6 correction to lunar longitude libration."""

    def correction_rad(
        self,
        epoch_tdb: Epoch,
        *,
        j2000_epoch_tdb: Epoch,
    ) -> float:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        j2000_epoch_tdb = require_tdb_epoch(
            j2000_epoch_tdb,
            name="j2000_epoch_tdb",
        )
        julian_centuries_since_j2000 = (
            (epoch_tdb.jd1 - j2000_epoch_tdb.jd1) + (epoch_tdb.jd2 - j2000_epoch_tdb.jd2)
        ) / _JULIAN_CENTURY_DAYS
        lunar_anomaly, solar_anomaly, argument_latitude, elongation, _ = (
            float(argument) for argument in _iers2010.fundarg(float(julian_centuries_since_j2000))
        )
        correction_mas = (
            4.5 * np.cos(solar_anomaly)
            + 1.8 * np.cos(2.0 * lunar_anomaly - 2.0 * elongation)
            + 10.5 * np.cos(2.0 * argument_latitude - 2.0 * lunar_anomaly)
        )
        return float(correction_mas * _MAS2RAD)


def make_longitude_libration_correction_model(
    correction_type: LongitudeLibrationCorrectionType | str | None,
) -> LongitudeLibrationCorrectionModel:
    normalized_type = normalize_longitude_libration_correction_type(correction_type)
    if normalized_type is LongitudeLibrationCorrectionType.NONE:
        return ZeroLongitudeLibrationCorrection()
    if normalized_type is LongitudeLibrationCorrectionType.INPOP21A:
        return Inpop21aLongitudeLibrationCorrection()
    raise AssertionError(f"Unhandled longitude-libration correction type: {normalized_type!r}")


__all__ = [
    "Inpop21aLongitudeLibrationCorrection",
    "LongitudeLibrationCorrectionModel",
    "ZeroLongitudeLibrationCorrection",
    "make_longitude_libration_correction_model",
    "normalize_longitude_libration_correction_type",
]
