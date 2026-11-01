"""Canonical normal-point validation at file import boundaries."""

import math

from lunarops.base.station_identity import station_id_from_ilrs_code
from lunarops.classes.observation.normal_points import NptRecord
from lunarops.classes.time import Epoch, TimeScale


def read_normal_point_record(**values) -> NptRecord:
    record = NptRecord(**values)
    if not isinstance(record.transmit_epoch, Epoch):
        raise TypeError("transmit_epoch must be an Epoch.")
    record.transmit_epoch.require_scale(TimeScale.UTC, name="transmit_epoch")
    record.station_name = _exact_identity(record.station_name)
    record.reflector_name = _exact_identity(record.reflector_name)
    if not record.station_name or not record.reflector_name:
        raise ValueError("station_name and reflector_name must not be empty.")
    positive_fields = (
        "round_trip_time_s",
        "uncertainty_two_way_s",
        "pressure_hpa",
        "temperature_k",
        "wavelength_nm",
    )
    for name in positive_fields:
        value = float(getattr(record, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
        setattr(record, name, value)
    humidity = float(record.humidity_percent)
    if not math.isfinite(humidity) or not 0.0 <= humidity <= 100.0:
        raise ValueError("humidity_percent must be finite and in [0, 100].")
    record.humidity_percent = humidity
    if isinstance(record.index, bool) or not isinstance(record.index, int) or record.index < 0:
        raise ValueError("Normal-point index must be a non-negative integer.")
    record.index = int(record.index)
    record.station_code = _optional_text(record.station_code)
    record.reflector_code = _optional_text(record.reflector_code)
    if record.station_code is not None:
        station = station_id_from_ilrs_code(record.station_code)
        if record.station_name != station:
            raise ValueError("station_name contradicts station_code.")
    if record.reflector_code in {"0", "1", "2", "3", "4"}:
        reflector = ("APOLLO11", "LUNOKHOD1", "APOLLO14", "APOLLO15", "LUNOKHOD2")[int(record.reflector_code)]
        if record.reflector_name != reflector:
            raise ValueError("reflector_name contradicts reflector_code.")
    return record


def _optional_text(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Source codes must be non-empty exact strings or None.")
    return value


def _exact_identity(value: object) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("Normal-point identifiers must be exact strings.")
    return value
