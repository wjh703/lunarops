"""Additive one-way station range-bias parametrization."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from lunarops.base.array_validation import parameter_vector
from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.time import Epoch
from lunarops.config.registry import register
from lunarops.config.schema import ConfigSchema, field, sequence, string

from .base import Parametrization

_MODES = {"station", "station+interval"}


def _iso_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        raise TypeError(f"{field} must be a date, not a datetime.")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be written as YYYY-MM-DD.") from exc


@dataclass(frozen=True, slots=True)
class StationBiasInterval:
    station: str
    start: date
    end_exclusive: date | None
    name: str | None = None

    def __post_init__(self) -> None:
        start = _iso_date(self.start, "stationRangeBias interval start")
        end = (
            None
            if self.end_exclusive is None
            else _iso_date(
                self.end_exclusive,
                "stationRangeBias interval end_exclusive",
            )
        )
        if end is not None and end <= start:
            raise ValueError("stationRangeBias interval end_exclusive must be after start.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)
        object.__setattr__(self, "name", None if self.name is None else self.name.strip())

    @property
    def key(self) -> str:
        end = self.end_exclusive.isoformat() if self.end_exclusive else "present"
        return self.name or f"{self.station}_{self.start.isoformat()}_{end}"

    def active_at(self, epoch: Epoch) -> bool:
        epoch_date = date.fromisoformat(epoch.date_iso())
        return self.start <= epoch_date and (self.end_exclusive is None or epoch_date < self.end_exclusive)


def parse_station_bias_intervals(config_value: object) -> list[StationBiasInterval]:
    """Build intervals from the schema-validated list-of-mappings config."""
    if config_value is None:
        return []
    intervals = [
        StationBiasInterval(
            station=item["station"],
            start=item["start"],
            end_exclusive=item["endExclusive"],
            name=item.get("name"),
        )
        for item in config_value
    ]
    keys = [interval.key for interval in intervals]
    if len(set(keys)) != len(keys):
        raise ValueError("stationRangeBias interval keys must be unique.")
    return intervals


def active_station_bias_interval_keys(
    intervals: Sequence[StationBiasInterval],
    eq: ObservationEquation,
    *,
    requested: set[str] | None = None,
) -> tuple[str, ...]:
    station = eq.station_key
    if requested is not None and station not in requested:
        return ()
    return tuple(
        interval.key
        for interval in intervals
        if interval.station == station and interval.active_at(eq.transmit_epoch_utc)
    )


_STATION_BIAS_INTERVAL_SCHEMA = ConfigSchema(
    fields=(
        string("station", required=True, non_empty=True, allow_none=False),
        field("start", "time", required=True, allow_none=False),
        field("endExclusive", "time", required=True, allow_none=True),
        string("name", non_empty=True),
    )
)


@register(
    "parametrization",
    "stationRangeBias",
    schema=ConfigSchema(
        fields=(
            sequence("stations", item_kind="string", min_items=1, non_empty=True),
            string("per", default="station", choices=tuple(sorted(_MODES)), allow_none=False),
            sequence("intervals", item_kind="mapping", item_nested=_STATION_BIAS_INTERVAL_SCHEMA),
        ),
        type_name="stationRangeBias",
    ),
)
class StationRangeBiasParametrization(Parametrization):
    """Estimate one-way station biases by station or explicit interval."""

    def __init__(
        self,
        *,
        stations: Sequence[str] | None = None,
        per: str = "station",
        intervals: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        parsed_intervals = parse_station_bias_intervals(intervals)
        if per == "station" and intervals is not None:
            raise ValueError("stationRangeBias intervals require per='station+interval'.")
        if per == "station+interval" and not parsed_intervals:
            raise ValueError("stationRangeBias per='station+interval' requires intervals.")
        self.per = per
        self.intervals = parsed_intervals
        self.requested = None if stations is None else list(stations)
        if self.requested is not None and len(set(self.requested)) != len(self.requested):
            raise ValueError("stationRangeBias stations must be unique.")
        self.keys: list[str] = []
        self._index_by_key: dict[str, int] = {}
        self._names: list[ParameterName] = []
        self.values: dict[str, float] = {}

    @classmethod
    def from_config(cls, config: dict, context) -> StationRangeBiasParametrization:
        return cls(
            stations=config.get("stations"),
            per=config.get("per", "station"),
            intervals=config.get("intervals"),
        )

    def _station_key_for(self, eq: ObservationEquation) -> str | None:
        station = eq.station_key
        if self.requested is not None and station not in self.requested:
            return None
        return station

    def _active_keys_for(self, eq: ObservationEquation) -> tuple[str, ...]:
        if self.per == "station":
            key = self._station_key_for(eq)
            return () if key is None else (key,)
        return active_station_bias_interval_keys(
            self.intervals,
            eq,
            requested=None if self.requested is None else set(self.requested),
        )

    def setup(self, equations: Sequence[ObservationEquation], model_state) -> None:
        observed_stations = {equation.station_key for equation in equations}
        if self.requested is not None:
            missing = set(self.requested) - observed_stations
            if missing:
                raise ValueError(f"stationRangeBias requested station(s) have no observations: {sorted(missing)}.")
        if self.per == "station":
            self.keys = sorted(observed_stations if self.requested is None else self.requested)
        else:
            self.keys = sorted({key for equation in equations for key in self._active_keys_for(equation)})
        for key in self.keys:
            self.values.setdefault(key, 0.0)
        self._index_by_key = {key: index for index, key in enumerate(self.keys)}
        self._names = self._build_parameter_names()

    def _build_parameter_names(self) -> list[ParameterName]:
        if self.per == "station":
            return [ParameterName(key, "rangeBias") for key in self.keys]
        interval_by_key = {interval.key: interval for interval in self.intervals}
        names: list[ParameterName] = []
        for key in self.keys:
            interval = interval_by_key[key]
            end = "present" if interval.end_exclusive is None else interval.end_exclusive.isoformat()
            names.append(
                ParameterName(
                    interval.station,
                    "rangeBias",
                    "interval",
                    f"{interval.start.isoformat()}/{end}",
                )
            )
        return names

    def parameter_names(self) -> list[ParameterName]:
        return list(self._names)

    def reference_values(self) -> np.ndarray:
        return np.asarray([self.values.get(key, 0.0) for key in self.keys], dtype=float)

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        return [(self._index_by_key[key], 1.0) for key in self._active_keys_for(eq) if key in self._index_by_key]

    def reduce_observation(self, eq: ObservationEquation) -> float:
        return float(sum(self.values.get(key, 0.0) for key in self._active_keys_for(eq)))

    def apply_update(self, delta: np.ndarray) -> None:
        updates = parameter_vector(
            delta,
            expected_size=len(self.keys),
            name="stationRangeBias update",
        )
        for key, update in zip(self.keys, updates):
            self.values[key] = self.values.get(key, 0.0) + float(update)

    def state(self) -> dict[str, object]:
        return {
            "per": self.per,
            "intervals": [
                {
                    "station": interval.station,
                    "start": interval.start.isoformat(),
                    "end_exclusive": (None if interval.end_exclusive is None else interval.end_exclusive.isoformat()),
                    "name": interval.name,
                }
                for interval in self.intervals
            ],
            "values": {key: float(self.values.get(key, 0.0)) for key in self.keys},
        }


__all__ = [
    "StationBiasInterval",
    "StationRangeBiasParametrization",
    "active_station_bias_interval_keys",
    "parse_station_bias_intervals",
]
