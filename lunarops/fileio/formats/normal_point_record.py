"""Validate canonical normal-point records at file import boundaries."""

import math

from lunarops.base.station_identity import station_id_from_ilrs_code
from lunarops.classes.observation.normal_points import NptRecord
from lunarops.classes.time import Epoch, TimeScale


def read_normal_point_record(**values) -> NptRecord:
    """Construct and validate one canonical ``NptRecord``.

    The record is normalized in place: numeric fields are converted to float and ``index`` to int. Checks:

    - ``transmit_epoch`` is an ``Epoch`` in UTC.
    - ``station_name`` and ``reflector_name`` are non-empty strings without surrounding whitespace.
    - ``round_trip_time_s``, ``uncertainty_two_way_s``, ``pressure_hpa``, ``temperature_k`` and
      ``wavelength_nm`` are positive and finite; ``humidity_percent`` is finite and in [0, 100].
    - ``index`` is a non-negative integer (not a bool).
    - ``station_code`` and ``reflector_code`` are None or non-empty strings without surrounding whitespace.
    - A ``station_code`` resolves to ``station_name`` through the ILRS station-code table.
    - A ``reflector_code`` of ``0`` to ``4`` names APOLLO11, LUNOKHOD1, APOLLO14, APOLLO15 or LUNOKHOD2,
      which must equal ``reflector_name``; other codes are not checked.

    Parameters
    ----------
    **values
        ``NptRecord`` fields: ``station_name``, ``reflector_name``, ``transmit_epoch`` (UTC ground transmit
        epoch), ``round_trip_time_s`` (two-way light time, s), ``uncertainty_two_way_s`` (s),
        ``pressure_hpa`` (hPa), ``temperature_k`` (K), ``humidity_percent`` (%), ``wavelength_nm`` (nm),
        ``index``, ``station_code`` and ``reflector_code``.

    Returns
    -------
    NptRecord
        Validated record.

    Raises
    ------
    TypeError
        If ``transmit_epoch`` is not an ``Epoch``.
    ValueError
        If any other check fails, including an unknown ``station_code``.
    """
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
