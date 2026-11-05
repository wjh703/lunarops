"""Parametrization: LLR station ITRF coordinates and optional linear velocity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.observation.resolver import ObservationCatalogState
from lunarops.config.schema import ConfigSchema, boolean, sequence

from .base import Parametrization

_AXES = ("x", "y", "z")


class StationPositionParametrization(Parametrization):
    TYPE = "stationPosition"
    SCHEMA = ConfigSchema(
        fields=(
            sequence("stations", item_kind="string", unique=True, min_items=1, non_empty=True),
            sequence("fixedStations", item_kind="string", unique=True, non_empty=True),
            boolean("estimateVelocity", default=False),
        ),
        type_name="stationPosition",
    )

    def __init__(
        self,
        *,
        model_state: ObservationCatalogState,
        stations: Sequence[str] | None = None,
        fixed_stations: Sequence[str] | None = None,
        estimate_velocity: bool = False,
    ) -> None:
        requested = None if stations is None else list(stations)
        fixed = [] if fixed_stations is None else list(fixed_stations)
        self.requested = requested
        self.fixed = fixed
        self.estimate_velocity = bool(estimate_velocity)
        self.keys: list[str] = []
        self.names: list[ParameterName] = []
        self._model_state = model_state

    @classmethod
    def from_config(cls, config: dict, context, *, model_state) -> StationPositionParametrization:
        return cls(
            model_state=model_state,
            stations=config.get("stations"),
            fixed_stations=config.get("fixedStations"),
            estimate_velocity=config["estimateVelocity"],
        )

    @property
    def _block_size(self) -> int:
        return 6 if self.estimate_velocity else 3

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
        catalog = self._model_state.station_catalog
        observed = sorted(set(batch.station.tolist()))
        unknown_fixed = set(self.fixed) - set(catalog)
        if unknown_fixed:
            raise KeyError(f"stationPosition: unknown fixed station key(s) {sorted(unknown_fixed)}")
        if self.requested is None:
            candidates = [key for key in observed if key in catalog]
        else:
            missing = set(self.requested) - set(catalog)
            if missing:
                raise KeyError(f"stationPosition: unknown station key(s) {sorted(missing)}")
            unobserved = set(self.requested) - set(observed)
            if unobserved:
                raise ValueError(f"stationPosition requested station(s) have no observations: {sorted(unobserved)}")
            candidates = list(self.requested)
        fixed = set(self.fixed)
        self.keys = [key for key in candidates if key not in fixed]
        if not self.keys:
            raise ValueError("stationPosition has no estimated stations after removing fixedStations.")
        names: list[ParameterName] = []
        for key in self.keys:
            names.extend(ParameterName(key, f"position.{axis}") for axis in _AXES)
            if self.estimate_velocity:
                names.extend(ParameterName(key, f"velocity.{axis}") for axis in _AXES)
        self.names = names

        return self.names

    def x0(self) -> np.ndarray:
        catalog = self._model_state.station_catalog
        values: list[float] = []
        for key in self.keys:
            values.extend(float(value) for value in np.asarray(catalog[key].itrf_xyz_m, dtype=float))
            if self.estimate_velocity:
                values.extend(float(value) for value in np.asarray(catalog[key].itrf_velocity_m_per_year, dtype=float))
        return np.asarray(values, dtype=float)

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        for j, key in enumerate(self.keys):
            rows = batch.station == key
            start = self._block_size * j
            partial = batch.station_position_itrf[rows]
            out[rows, start : start + 3] += partial
            if self.estimate_velocity:
                record = self._model_state.station_catalog[key]
                if record.position_epoch_utc not in batch.year_columns:
                    batch.year_columns[record.position_epoch_utc] = np.asarray(
                        [record.years_since_position_epoch(batch.epoch(i)) for i in range(len(batch))]
                    )
                years = batch.year_columns[record.position_epoch_utc][rows]
                out[rows, start + 3 : start + 6] += partial * years[:, None]

    def reduce(self, batch: EquationBatch) -> np.ndarray:
        return np.zeros(len(batch))

    def update(self, delta: np.ndarray) -> None:
        size = self._block_size
        delta = np.asarray(delta, float)
        state = self._model_state
        updates = {}
        for j, key in enumerate(self.keys):
            record = state.station_catalog[key]
            block = delta[size * j : size * (j + 1)]
            position = np.asarray(record.itrf_xyz_m, dtype=float) + block[:3]
            velocity = np.asarray(record.itrf_velocity_m_per_year, dtype=float)
            if self.estimate_velocity:
                velocity = velocity + block[3:6]
            updates[key] = (position, velocity)
        state.apply_station_states_itrf_m(updates)

    def max_update_norm(self, delta: np.ndarray) -> float:
        # Convergence is judged on the position update; velocities are m/yr.
        size = self._block_size
        if not len(delta):
            return 0.0
        return max(float(np.linalg.norm(delta[size * j : size * j + 3])) for j in range(len(self.keys)))

    def state(self) -> dict[str, object]:
        catalog = self._model_state.station_catalog
        result: dict[str, object] = {}
        for key in self.keys:
            entry: dict[str, object] = {
                "position": [float(value) for value in np.asarray(catalog[key].itrf_xyz_m, dtype=float)]
            }
            if self.estimate_velocity:
                entry["velocity"] = [
                    float(value) for value in np.asarray(catalog[key].itrf_velocity_m_per_year, dtype=float)
                ]
            result[key] = entry
        result["fixedStations"] = list(self.fixed)
        return result
