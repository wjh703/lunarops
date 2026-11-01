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
    """Estimate corrections to station ITRF reference coordinates.

    Options
    -------
    stations :
        Explicit station catalog keys to estimate; default is every observed
        station.
    fixedStations :
        Datum stations held at their catalog values. They are removed from the
        estimated set, so at least one well-observed station fixes the network
        frame against reflector coordinates and range biases.
    estimateVelocity :
        Also estimate the ITRF velocity (m/yr). The position parameter refers
        to the catalog ``position_epoch_utc``.
    """

    TYPE = "stationPosition"
    SCHEMA = ConfigSchema(
        fields=(
            sequence("stations", item_kind="string", min_items=1, non_empty=True),
            sequence("fixedStations", item_kind="string", non_empty=True),
            boolean("estimateVelocity", default=False),
        ),
        type_name="stationPosition",
    )

    def __init__(
        self,
        *,
        stations: Sequence[str] | None = None,
        fixed_stations: Sequence[str] | None = None,
        estimate_velocity: bool = False,
    ) -> None:
        requested = None if stations is None else list(stations)
        if requested is not None and len(set(requested)) != len(requested):
            raise ValueError("stationPosition station keys must be unique.")
        fixed = [] if fixed_stations is None else list(fixed_stations)
        if len(set(fixed)) != len(fixed):
            raise ValueError("stationPosition fixedStations must be unique.")
        self.requested = requested
        self.fixed = fixed
        self.estimate_velocity = bool(estimate_velocity)
        self.keys: list[str] = []
        self._index_by_key: dict[str, int] = {}
        self._names: list[ParameterName] = []
        self._model_state: ObservationCatalogState | None = None

    @classmethod
    def from_config(cls, config: dict, context) -> StationPositionParametrization:
        return cls(
            stations=config.get("stations"),
            fixed_stations=config.get("fixedStations"),
            estimate_velocity=bool(config.get("estimateVelocity", False)),
        )

    @property
    def _block_size(self) -> int:
        return 6 if self.estimate_velocity else 3

    def setup(
        self,
        equations: Sequence[EquationBatch],
        model_state: ObservationCatalogState,
    ) -> None:
        self._model_state = model_state
        catalog = model_state.station_catalog
        observed = sorted({eq.station[0] for eq in equations})
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
        self._index_by_key = {key: index for index, key in enumerate(self.keys)}
        names: list[ParameterName] = []
        for key in self.keys:
            names.extend(ParameterName(key, f"position.{axis}") for axis in _AXES)
            if self.estimate_velocity:
                names.extend(ParameterName(key, f"velocity.{axis}") for axis in _AXES)
        self._names = names

    def parameter_names(self) -> list[ParameterName]:
        return list(self._names)

    def _require_state(self) -> ObservationCatalogState:
        if self._model_state is None:
            raise RuntimeError("stationPosition has not been set up.")
        return self._model_state

    def reference_values(self) -> np.ndarray:
        catalog = self._require_state().station_catalog
        values: list[float] = []
        for key in self.keys:
            values.extend(float(value) for value in np.asarray(catalog[key].itrf_xyz_m, dtype=float))
            if self.estimate_velocity:
                values.extend(float(value) for value in np.asarray(catalog[key].itrf_velocity_m_per_year, dtype=float))
        return np.asarray(values, dtype=float)

    def design_entries(self, eq: EquationBatch) -> list[tuple[int, float]]:
        j = self._index_by_key.get(eq.station[0])
        if j is None:
            return []
        block = eq.station_position_itrf[0]
        if block is None:
            raise KeyError(
                "Observation equation lacks the 'station_position_itrf' partial block; "
                "run the forward model with include_station_position_partials=True."
            )
        partial = np.asarray(block, dtype=float).reshape(3)
        start = self._block_size * j
        entries = [(start + axis, float(value)) for axis, value in enumerate(partial) if float(value)]
        if self.estimate_velocity:
            years = (
                self._require_state().station_catalog[eq.station[0]].years_since_position_epoch(eq.epoch(0))
            )
            entries.extend(
                (start + 3 + axis, float(value) * years) for axis, value in enumerate(partial) if float(value)
            )
        return entries

    def apply_update(self, delta: np.ndarray) -> None:
        size = self._block_size
        delta = np.asarray(delta, float)
        state = self._require_state()
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
        if self._model_state is None:
            return {}
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
