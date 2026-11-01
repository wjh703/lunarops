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
            sequence("stations", item_kind="string", min_items=1, non_empty=True),
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

    def _station_key_for(self, eq: EquationBatch) -> str | None:
        station = eq.station[0]
        if self.requested is not None and station not in self.requested:
            return None
        return station

    def _active_keys_for(self, eq: EquationBatch) -> tuple[str, ...]:
        if self.per == "station":
            key = self._station_key_for(eq)
            return () if key is None else (key,)
        return active_station_bias_interval_keys(
            self.intervals,
            eq,
            requested=None if self.requested is None else set(self.requested),
        )

    def setup(self, equations: Sequence[EquationBatch], model_state) -> None:
        observed_stations = {equation.station[0] for equation in equations}
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

    def design_entries(self, eq: EquationBatch) -> list[tuple[int, float]]:
        return [(self._index_by_key[key], 1.0) for key in self._active_keys_for(eq) if key in self._index_by_key]

    def reduce_observation(self, eq: EquationBatch) -> float:
        return float(sum(self.values.get(key, 0.0) for key in self._active_keys_for(eq)))

    def apply_update(self, delta: np.ndarray) -> None:
        updates = np.asarray(delta, float)
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
    "StationRangeBiasParametrization",
    "active_station_bias_interval_keys",
    "parse_station_bias_intervals",
]
