"""Lunar body-fixed orientation and optional longitude-libration correction."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.time import Epoch, require_tdb_epoch

_MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)
_JULIAN_CENTURY_DAYS = 36_525.0


class LongitudeLibrationCorrection(StrEnum):
    NONE = "none"
    INPOP21A = "inpop21a"


@runtime_checkable
class LunarOrientationProvider(Protocol):
    def pa_to_lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray: ...

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float: ...

    @property
    def correction(self) -> LongitudeLibrationCorrection: ...


def normalize_longitude_libration_correction(
    value: LongitudeLibrationCorrection | str | None,
) -> LongitudeLibrationCorrection:
    if isinstance(value, LongitudeLibrationCorrection):
        return value
    if value is None:
        return LongitudeLibrationCorrection.NONE
    if not isinstance(value, str):
        raise TypeError("longitude-libration correction must be a string, enum, or None")
    try:
        return LongitudeLibrationCorrection(value.strip().lower())
    except ValueError:
        allowed = ", ".join(item.value for item in LongitudeLibrationCorrection)
        raise ValueError(f"Unsupported longitude-libration correction {value!r}; expected one of: {allowed}.") from None


def _passive_rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(((cosine, sine, 0.0), (-sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _passive_rotation_x(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(((1.0, 0.0, 0.0), (0.0, cosine, sine), (0.0, -sine, cosine)))


class CalcephLunarOrientation:
    def __init__(
        self,
        angles_at_epoch: Callable[[Epoch], np.ndarray],
        correction: LongitudeLibrationCorrection | str | None = None,
        *,
        angles_many_at_epochs: Callable[[tuple[Epoch, ...]], np.ndarray] | None = None,
        j2000_epoch_tdb: Epoch | None = None,
    ) -> None:
        self._angles_at_epoch = angles_at_epoch
        self._angles_many_at_epochs = angles_many_at_epochs
        self._correction = normalize_longitude_libration_correction(correction)
        self._j2000_epoch_tdb = (
            None
            if j2000_epoch_tdb is None
            else require_tdb_epoch(j2000_epoch_tdb, name="j2000_epoch_tdb")
        )

    @property
    def correction(self) -> LongitudeLibrationCorrection:
        return self._correction

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        if self._correction is LongitudeLibrationCorrection.NONE:
            return 0.0
        if self._j2000_epoch_tdb is None:
            raise RuntimeError("INPOP21A longitude correction requires a J2000 TDB epoch.")
        centuries = ((epoch.jd1 - self._j2000_epoch_tdb.jd1) + (epoch.jd2 - self._j2000_epoch_tdb.jd2)) / _JULIAN_CENTURY_DAYS
        lunar_anomaly, solar_anomaly, argument_latitude, elongation, _ = (
            float(value) for value in _iers2010.fundarg(float(centuries))
        )
        correction_mas = (
            4.5 * np.cos(solar_anomaly)
            + 1.8 * np.cos(2.0 * lunar_anomaly - 2.0 * elongation)
            + 10.5 * np.cos(2.0 * argument_latitude - 2.0 * lunar_anomaly)
        )
        return float(correction_mas * _MAS_TO_RAD)

    def pa_to_lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        phi, theta, psi = np.asarray(self._angles_at_epoch(epoch), dtype=float)[:3]
        psi += self.longitude_libration_correction_rad(epoch)
        matrix = (_passive_rotation_z(psi) @ _passive_rotation_x(theta) @ _passive_rotation_z(phi)).T
        result = np.ascontiguousarray(matrix)
        result.setflags(write=False)
        return result

    def pa_to_lcrs_matrices(self, epochs) -> np.ndarray:
        epoch_array = tuple(require_tdb_epoch(epoch, name="epoch_tdb") for epoch in epochs)
        if not epoch_array:
            return np.empty((0, 3, 3), dtype=float)
        if self._angles_many_at_epochs is None:
            return np.asarray([self.pa_to_lcrs_matrix(epoch) for epoch in epoch_array], dtype=float)
        angles = np.asarray(self._angles_many_at_epochs(epoch_array), dtype=float)
        if angles.shape != (len(epoch_array), 3):
            raise ValueError("Batch lunar orientation provider returned an invalid angle matrix")
        matrices = np.empty((len(epoch_array), 3, 3), dtype=float)
        for index, (epoch, values) in enumerate(zip(epoch_array, angles, strict=True)):
            phi, theta, psi = values
            psi += self.longitude_libration_correction_rad(epoch)
            matrices[index] = (_passive_rotation_z(psi) @ _passive_rotation_x(theta) @ _passive_rotation_z(phi)).T
        return matrices


class FixedLunarOrientation:
    def __init__(
        self,
        matrix_at_epoch: Callable[[Epoch], np.ndarray],
        correction=LongitudeLibrationCorrection.NONE,
        correction_at_epoch: Callable[[Epoch], float] | None = None,
    ):
        self._matrix_at_epoch = matrix_at_epoch
        self._correction = normalize_longitude_libration_correction(correction)
        self._correction_at_epoch = correction_at_epoch

    @property
    def correction(self) -> LongitudeLibrationCorrection:
        return self._correction

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return 0.0 if self._correction_at_epoch is None else float(self._correction_at_epoch(epoch))

    def pa_to_lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        return self._matrix_at_epoch(require_tdb_epoch(epoch_tdb, name="epoch_tdb"))

    def pa_to_lcrs_matrices(self, epochs) -> np.ndarray:
        return np.asarray([self.pa_to_lcrs_matrix(epoch) for epoch in epochs], dtype=float)


__all__ = [
    "CalcephLunarOrientation",
    "FixedLunarOrientation",
    "LongitudeLibrationCorrection",
    "LunarOrientationProvider",
    "normalize_longitude_libration_correction",
]
