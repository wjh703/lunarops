"""Parametrization of LLR station ITRF coordinates and optional linear velocities."""

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
    """LLR station ITRF reference coordinates, three parameters per station, plus optional linear velocities.

    The parameters are the ITRF X, Y, Z coordinates at each station's reference epoch, in metres
    (``<station>:position.<axis>``), and with ``estimate_velocity`` the ITRF velocity components in metres per
    Julian year (``<station>:velocity.<axis>``). Updates are written to the shared station catalog, so the
    forward model is re-evaluated with them.

    Parameters
    ----------
    model_state : ObservationCatalogState
        Shared catalogs holding the station states.
    stations : Sequence[str], optional
        Station keys to estimate. ``None`` estimates every observed station present in the catalog.
    fixed_stations : Sequence[str], optional
        Station keys removed from the estimated set and held at their catalog values.
    estimate_velocity : bool, default False
        Also estimate the linear ITRF velocity of each station.
    """

    TYPE = "stationPosition"
    SCHEMA = ConfigSchema(
        fields=(
            sequence(
                "stations",
                item_kind="string",
                unique=True,
                min_items=1,
                non_empty=True,
                description="Station catalog keys to estimate. Defaults to every observed station in the catalog.",
            ),
            sequence(
                "fixedStations",
                item_kind="string",
                unique=True,
                non_empty=True,
                description="Station keys held at their catalog positions.",
            ),
            boolean(
                "estimateVelocity",
                default=False,
                description="Also estimate the linear ITRF velocity of each station.",
            ),
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
        """Create the parametrization from a validated configuration."""
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
        """Select the estimated stations and return their parameter names.

        Raises
        ------
        KeyError
            If a requested or fixed station is not in the catalog.
        ValueError
            If a requested station has no observations in ``batch`` or no station remains after removing the
            fixed stations.
        """
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
        """Return the current catalog ITRF positions (m) and, if estimated, velocities (m per Julian year)."""
        catalog = self._model_state.station_catalog
        values: list[float] = []
        for key in self.keys:
            values.extend(float(value) for value in np.asarray(catalog[key].itrf_xyz_m, dtype=float))
            if self.estimate_velocity:
                values.extend(float(value) for value in np.asarray(catalog[key].itrf_velocity_m_per_year, dtype=float))
        return np.asarray(values, dtype=float)

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        """Add the station design columns of each station's rows.

        Position columns are the ``station_position_itrf`` partials (m/m). Velocity columns are the same partials
        multiplied by the Julian years from the station reference epoch to the transmit epoch (m per m/yr).
        """
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
        """Return zeros: station states enter the forward model, not the reduced observations."""
        return np.zeros(len(batch))

    def update(self, delta: np.ndarray) -> None:
        """Add the position (m) and velocity (m per Julian year) corrections to the shared station catalog."""
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
        """Return the largest 3-D position correction of any station, in metres; velocities are not included."""
        # Convergence is judged on the position update; velocities are m/yr.
        size = self._block_size
        if not len(delta):
            return 0.0
        return max(float(np.linalg.norm(delta[size * j : size * j + 3])) for j in range(len(self.keys)))

    def state(self) -> dict[str, object]:
        """Return the current ITRF position (and velocity) per estimated station and the fixed-station list."""
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
