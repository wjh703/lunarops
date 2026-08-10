"""Observation-level range-bias models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from lunarops.base.constants import C
from lunarops.classes.time import Epoch, TimeScale

from .table import AdditiveRangeBiasTable, RangeBiasLookup, RangeBiasLookupStatus


@dataclass(frozen=True, slots=True)
class RangeBiasRequest:
    """Validated inputs for one range-bias evaluation."""

    station_identifiers: tuple[str, ...]
    observation_epoch_utc: Epoch

    def __post_init__(self) -> None:
        if not isinstance(self.station_identifiers, tuple) or not self.station_identifiers:
            raise TypeError("station_identifiers must be a non-empty tuple of strings.")
        if any(not isinstance(identifier, str) or not identifier.strip() for identifier in self.station_identifiers):
            raise TypeError("station_identifiers must contain non-empty strings.")
        if not isinstance(self.observation_epoch_utc, Epoch):
            raise TypeError("observation_epoch_utc must be an Epoch.")
        self.observation_epoch_utc.require_scale(TimeScale.UTC, name="observation_epoch_utc")
        object.__setattr__(
            self, "station_identifiers", tuple(identifier.strip() for identifier in self.station_identifiers)
        )


@dataclass(frozen=True, slots=True)
class RangeBiasCorrection:
    """A range-bias lookup and its derived light-time corrections."""

    model_label: str
    lookup: RangeBiasLookup

    @property
    def correction_two_way_cm(self) -> float:
        return self.lookup.correction_two_way_cm

    @property
    def correction_two_way_m(self) -> float:
        return 0.01 * self.correction_two_way_cm

    @property
    def correction_round_trip_time_s(self) -> float:
        return self.correction_two_way_m / C

    @property
    def correction_one_way_m(self) -> float:
        return 0.5 * self.correction_two_way_m

    def apply_to_computed_round_trip_time_s(self, computed_round_trip_time_s: float) -> float:
        return float(computed_round_trip_time_s) - self.correction_round_trip_time_s


class RangeBiasModel(ABC):
    @abstractmethod
    def evaluate(self, request: RangeBiasRequest) -> RangeBiasCorrection:
        """Evaluate one range-bias request."""


class ZeroRangeBiasModel(RangeBiasModel):
    model_label = "none"

    def evaluate(self, request: RangeBiasRequest) -> RangeBiasCorrection:
        day = date.fromisoformat(request.observation_epoch_utc.date_iso())
        lookup = RangeBiasLookup(
            requested_station_identifiers=request.station_identifiers,
            matched_station_id=None,
            observation_date_utc=day,
            active_components=(),
            status=RangeBiasLookupStatus.EXPLICIT_ZERO,
        )
        return RangeBiasCorrection(self.model_label, lookup)


class TableRangeBiasModel(RangeBiasModel):
    def __init__(self, bias_table: AdditiveRangeBiasTable) -> None:
        self.bias_table = bias_table

    @property
    def model_label(self) -> str:
        return self.bias_table.source or "additive range-bias table"

    def evaluate(self, request: RangeBiasRequest) -> RangeBiasCorrection:
        lookup = self.bias_table.lookup(request.station_identifiers, request.observation_epoch_utc)
        return RangeBiasCorrection(self.model_label, lookup)


__all__ = [
    "RangeBiasCorrection",
    "RangeBiasModel",
    "RangeBiasRequest",
    "TableRangeBiasModel",
    "ZeroRangeBiasModel",
]
