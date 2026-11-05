"""Catalog resolution for source-independent normal-point observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from lunarops.classes.time import Epoch

from .catalogs import ReflectorRecord, StationRecord, resolve_catalog_key
from .normal_points import NptRecord


@dataclass(frozen=True, slots=True)
class ObservationCatalogSelection:
    """Catalog keys that override the station and reflector names of every normal point.

    Attributes
    ----------
    station_identifier : str or None
        Station catalog key, or ``None`` to use each record's ``station_name``.
    reflector_identifier : str or None
        Reflector catalog key, or ``None`` to use each record's ``reflector_name``.
    """

    station_identifier: str | None = None
    reflector_identifier: str | None = None


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ResolvedObservation:
    """Normal point with its resolved station and reflector records.

    Attributes
    ----------
    normal_point : NptRecord
        The normal point.
    station_key : str
        Station catalog key.
    station : StationRecord
        Station record at resolution time.
    reflector_key : str
        Reflector catalog key.
    reflector : ReflectorRecord
        Reflector record at resolution time.
    """

    normal_point: NptRecord
    station_key: str
    station: StationRecord
    reflector_key: str
    reflector: ReflectorRecord

    @property
    def transmit_epoch_utc(self) -> Epoch:
        """Transmit epoch of the normal point, UTC."""
        return self.normal_point.transmit_epoch


class ObservationCatalogState:
    """Mutable station and reflector catalogs shared by the observation model and the parametrizations.

    Parametrizations write updated coordinates here, so later observation evaluations use them.

    Parameters
    ----------
    station_catalog : Mapping[str, StationRecord]
        Station records by key; copied into a new dict.
    reflector_catalog : Mapping[str, ReflectorRecord]
        Reflector records by key; copied into a new dict.
    """

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
        """Return the reflector Moon PA positions (x, y, z) per key, in metres."""
        positions: dict[str, tuple[float, float, float]] = {}
        for key, record in self.reflector_catalog.items():
            values = np.asarray(record.moon_fixed_xyz_m, dtype=np.float64).reshape(3)
            positions[key] = (float(values[0]), float(values[1]), float(values[2]))
        return positions

    def apply_reflector_positions_pa_m(
        self,
        positions_pa_m_by_key: Mapping[str, Sequence[float]],
    ) -> None:
        """Replace reflector Moon PA positions.

        Parameters
        ----------
        positions_pa_m_by_key : Mapping[str, Sequence[float]]
            New position (x, y, z) in the Moon PA frame, in metres, per reflector key.

        Raises
        ------
        KeyError
            If a key is not in the reflector catalog.
        """
        unknown = set(positions_pa_m_by_key) - set(self.reflector_catalog)
        if unknown:
            raise KeyError(f"Unknown reflector state key(s): {sorted(unknown)}")
        for key, values in positions_pa_m_by_key.items():
            position = np.asarray(values, float)
            self.reflector_catalog[key] = replace(
                self.reflector_catalog[key],
                moon_fixed_xyz_m=position,
            )


    def station_states_itrf_m(self) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Return the ITRF reference position (m) and velocity (m per Julian year) per station key."""
        states = {}
        for key, record in self.station_catalog.items():
            position = np.asarray(record.itrf_xyz_m, dtype=np.float64).reshape(3)
            velocity = np.asarray(record.itrf_velocity_m_per_year, dtype=np.float64).reshape(3)
            states[key] = (
                (float(position[0]), float(position[1]), float(position[2])),
                (float(velocity[0]), float(velocity[1]), float(velocity[2])),
            )
        return states

    def apply_station_states_itrf_m(
        self,
        states_by_key: Mapping[str, tuple[ArrayLike, ArrayLike]],
    ) -> None:
        """Replace station ITRF reference positions and velocities.

        Parameters
        ----------
        states_by_key : Mapping[str, tuple[ArrayLike, ArrayLike]]
            New ``(position, velocity)`` per station key: ITRF position in metres at the record's reference
            epoch and ITRF velocity in metres per Julian year.

        Raises
        ------
        KeyError
            If a key is not in the station catalog.
        """
        unknown = set(states_by_key) - set(self.station_catalog)
        if unknown:
            raise KeyError(f"Unknown station state key(s): {sorted(unknown)}")
        for key, (position_values, velocity_values) in states_by_key.items():
            self.station_catalog[key] = replace(
                self.station_catalog[key],
                itrf_xyz_m=np.asarray(position_values, float),
                itrf_velocity_m_per_year=np.asarray(velocity_values, float),
            )


class ObservationResolver:
    """Resolve normal points against the station and reflector catalogs.

    Parameters
    ----------
    model_state : ObservationCatalogState
        Catalogs to resolve against.
    """

    def __init__(
        self,
        model_state: ObservationCatalogState,
    ) -> None:
        self.model_state = model_state

    def resolve(
        self,
        normal_point: NptRecord,
        catalog_selection: ObservationCatalogSelection | None = None,
    ) -> ResolvedObservation:
        """Resolve one normal point.

        Parameters
        ----------
        normal_point : NptRecord
            Normal point; its station and reflector names are used as catalog keys unless overridden.
        catalog_selection : ObservationCatalogSelection, optional
            Catalog key overrides.

        Returns
        -------
        ResolvedObservation
            The normal point with its catalog records.

        Raises
        ------
        KeyError
            If the station or reflector key is not in its catalog.
        """
        catalog_selection = catalog_selection or ObservationCatalogSelection()
        station_key = resolve_catalog_key(
            catalog_selection.station_identifier if catalog_selection.station_identifier is not None else normal_point.station_name,
            self.model_state.station_catalog, "Station",
        )
        reflector_key = resolve_catalog_key(
            catalog_selection.reflector_identifier if catalog_selection.reflector_identifier is not None else normal_point.reflector_name,
            self.model_state.reflector_catalog, "Reflector",
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
        """Resolve every normal point, reporting all failures together.

        Parameters
        ----------
        normal_points : Sequence[NptRecord]
            Normal points in order.
        catalog_selection : ObservationCatalogSelection, optional
            Catalog key overrides.

        Returns
        -------
        list[ResolvedObservation]
            Resolved observations in input order.

        Raises
        ------
        ValueError
            If any record cannot be resolved; the message lists each failing record position.
        """
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
