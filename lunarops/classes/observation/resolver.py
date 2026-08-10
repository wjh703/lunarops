"""Catalog resolution for source-independent normal-point observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from lunarops.base.array_validation import catalog_vector3
from lunarops.classes.time import Epoch, TimeScale
from .catalogs import ReflectorRecord, StationRecord, first_resolvable_key
from .normal_points import NptRecord


@dataclass(frozen=True, slots=True)
class ObservationCatalogSelection:
    station_identifier: str | None = None
    reflector_identifier: str | None = None


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ResolvedObservation:
    normal_point: NptRecord
    station_key: str
    station: StationRecord
    reflector_key: str
    reflector: ReflectorRecord

    @property
    def transmit_epoch_utc(self) -> Epoch:
        epoch = self.normal_point.transmit_epoch
        epoch.require_scale(TimeScale.UTC, name="normal_point.transmit_epoch")
        return epoch

    @property
    def station_identity_candidates(self) -> tuple[str, ...]:
        values = (
            self.station_key,
            self.station.name,
            self.normal_point.station_name,
            self.normal_point.station_code,
        )
        return tuple(value for value in values if value is not None and value.strip())


class ObservationCatalogState:
    """Mutable catalog state shared explicitly by model and parametrizations."""

    station_catalog: dict[str, StationRecord]
    reflector_catalog: dict[str, ReflectorRecord]

    __slots__ = ("reflector_catalog", "station_catalog")

    def __init__(
        self,
        station_catalog: Mapping[str, StationRecord],
        reflector_catalog: Mapping[str, ReflectorRecord],
    ) -> None:
        self.station_catalog = dict(station_catalog)
        self.reflector_catalog = dict(reflector_catalog)

    def reflector_positions_pa_m(self) -> dict[str, tuple[float, float, float]]:
        positions: dict[str, tuple[float, float, float]] = {}
        for key, record in self.reflector_catalog.items():
            values = np.asarray(record.moon_fixed_xyz_m, dtype=np.float64).reshape(3)
            positions[key] = (float(values[0]), float(values[1]), float(values[2]))
        return positions

    def apply_reflector_positions_pa_m(
        self,
        positions_pa_m_by_key: Mapping[str, Sequence[float]],
    ) -> None:
        unknown = set(positions_pa_m_by_key) - set(self.reflector_catalog)
        if unknown:
            raise KeyError(f"Unknown reflector state key(s): {sorted(unknown)}")
        for key, values in positions_pa_m_by_key.items():
            position = catalog_vector3(values, name=f"reflector[{key}].moon_fixed_xyz_m")
            self.reflector_catalog[key] = replace(
                self.reflector_catalog[key],
                moon_fixed_xyz_m=position,
            )


class ObservationResolver:
    def __init__(
        self,
        model_state: ObservationCatalogState,
    ) -> None:
        if not isinstance(model_state, ObservationCatalogState):
            raise TypeError("model_state must be an ObservationCatalogState.")
        self.model_state = model_state

    @staticmethod
    def _candidates(
        normal_point: NptRecord,
        catalog_selection: ObservationCatalogSelection,
    ) -> tuple[list[str | None], list[str | None]]:
        station_candidates: list[str | None] = (
            [catalog_selection.station_identifier]
            if catalog_selection.station_identifier
            else [normal_point.station_name, normal_point.station_code]
        )
        reflector_candidates: list[str | None] = (
            [catalog_selection.reflector_identifier]
            if catalog_selection.reflector_identifier
            else [normal_point.reflector_name, normal_point.reflector_code]
        )
        return station_candidates, reflector_candidates

    def resolve(
        self,
        normal_point: NptRecord,
        catalog_selection: ObservationCatalogSelection | None = None,
    ) -> ResolvedObservation:
        catalog_selection = catalog_selection or ObservationCatalogSelection()
        station_candidates, reflector_candidates = self._candidates(normal_point, catalog_selection)
        station_key = first_resolvable_key(station_candidates, self.model_state.station_catalog, "Station")
        reflector_key = first_resolvable_key(
            reflector_candidates,
            self.model_state.reflector_catalog,
            "Reflector",
        )
        return ResolvedObservation(
            normal_point=normal_point,
            station_key=station_key,
            station=self.model_state.station_catalog[station_key],
            reflector_key=reflector_key,
            reflector=self.model_state.reflector_catalog[reflector_key],
        )

    def resolve_all(
        self,
        normal_points: Sequence[NptRecord],
        catalog_selection: ObservationCatalogSelection | None = None,
    ) -> list[ResolvedObservation]:
        resolved: list[ResolvedObservation] = []
        problems: list[str] = []
        for position, normal_point in enumerate(normal_points):
            try:
                resolved.append(self.resolve(normal_point, catalog_selection))
            except KeyError as exc:
                problems.append(f"record_index={position}: {exc}")
        if problems:
            detail = "\n  ".join(problems)
            raise ValueError(f"Catalog resolution failed for {len(problems)} record(s):\n  {detail}")
        return resolved


__all__ = [
    "ObservationCatalogSelection",
    "ObservationCatalogState",
    "ObservationResolver",
    "ResolvedObservation",
]
