"""Observation-level range-bias models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from lunarops.base.constants import C
from lunarops.classes.time import Epoch

from .table import AdditiveRangeBiasTable, RangeBiasLookup, RangeBiasLookupStatus


@dataclass(frozen=True, slots=True)
class RangeBiasRequest:
    """Inputs for one range-bias evaluation."""

    station_id: str
    observation_epoch_utc: Epoch


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
            requested_station_id=request.station_id,
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
        lookup = self.bias_table.lookup(request.station_id, request.observation_epoch_utc)
        return RangeBiasCorrection(self.model_label, lookup)


__all__ = [
    "RangeBiasCorrection",
    "RangeBiasModel",
    "RangeBiasRequest",
    "TableRangeBiasModel",
    "ZeroRangeBiasModel",
]
