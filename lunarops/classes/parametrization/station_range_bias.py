"""Additive one-way station range-bias parametrization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.base.station_interval import StationInterval
from lunarops.classes.observation.equations import EquationBatch
from lunarops.config.schema import ConfigSchema, field, sequence, string

from .base import Parametrization

_MODES = {"station", "station+interval"}


def parse_station_bias_intervals(config_value: Sequence[Mapping[str, Any]] | None) -> list[StationInterval]:
    if config_value is None:
        return []
    intervals = [
        StationInterval(
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
    intervals: Sequence[StationInterval],
    eq: EquationBatch,
    *,
    requested: set[str] | None = None,
) -> tuple[str, ...]:
    station = eq.station[0]
    if requested is not None and station not in requested:
        return ()
    return tuple(
        interval.key
        for interval in intervals
        if interval.contains(station, StationInterval.parse(eq.epoch(0).date_iso()))
    )


_STATION_BIAS_INTERVAL_SCHEMA = ConfigSchema(
    fields=(
        string("station", required=True, non_empty=True),
        field("start", "time", required=True),
        field("endExclusive", "time", required=True, allow_none=True),
        string("name", non_empty=True),
    )
)


class StationRangeBiasParametrization(Parametrization):
    """Estimate one-way station biases by station or explicit interval."""

    TYPE = "stationRangeBias"
    SCHEMA = ConfigSchema(
        fields=(
            sequence("stations", item_kind="string", unique=True, min_items=1, non_empty=True),
            string("per", default="station", choices=tuple(sorted(_MODES))),
            sequence("intervals", item_kind="mapping", item_nested=_STATION_BIAS_INTERVAL_SCHEMA),
        ),
        type_name="stationRangeBias",
    )

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
        self.keys: list[str] = []
        self.names: list[ParameterName] = []
        self.values: dict[str, float] = {}

    @classmethod
    def from_config(cls, config: dict, context) -> StationRangeBiasParametrization:
        return cls(
            stations=config.get("stations"),
            per=config.get("per", "station"),
            intervals=config.get("intervals"),
        )

    def _masks(self, batch: EquationBatch):
        stations = batch.station
        if self.per == "station":
            return {key: stations == key for key in self.keys}
        days = np.asarray([StationInterval.parse(batch.epoch(i).date_iso()) for i in range(len(batch))])
        return {interval.key: interval.contains(stations, days)
                for interval in self.intervals
                if self.requested is None or interval.station in self.requested}

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
        observed_stations = set(batch.station.tolist())
        if self.requested is not None:
            missing = set(self.requested) - observed_stations
            if missing:
                raise ValueError(f"stationRangeBias requested station(s) have no observations: {sorted(missing)}.")
        self.keys = sorted(observed_stations if self.requested is None else self.requested)
        if self.per != "station":
            self.keys = sorted(key for key, mask in self._masks(batch).items() if np.any(mask))
        for key in self.keys:
            self.values.setdefault(key, 0.0)
        self.names = self._build_parameter_names()
        return self.names

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

    def x0(self) -> np.ndarray:
        return np.asarray([self.values[key] for key in self.keys])

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        masks = self._masks(batch)
        for j, key in enumerate(self.keys):
            out[:, j] = masks[key]

    def _contributions(self, batch: EquationBatch):
        for key, mask in self._masks(batch).items():
            yield np.where(mask, self.values.get(key, 0.0), 0.0)

    def update(self, delta: np.ndarray) -> None:
        for key, value in zip(self.keys, delta):
            self.values[key] += float(value)

    def state(self) -> dict[str, object]:
        return {
            "per": self.per,
            "intervals": [
                {
                    "station": interval.station,
                    "start": interval.start.isoformat(),
                    "endExclusive": (None if interval.end_exclusive is None else interval.end_exclusive.isoformat()),
                    "name": interval.name,
                }
                for interval in self.intervals
            ],
            "values": {key: float(self.values.get(key, 0.0)) for key in self.keys},
        }


__all__ = [
    "StationRangeBiasParametrization",
    "active_station_bias_interval_keys",
    "parse_station_bias_intervals",
]
