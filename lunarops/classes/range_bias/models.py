"""Observation-level range-bias models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

from lunarops.base.constants import C
from lunarops.classes.range_bias.table import RANGE_BIAS_ROW_SCHEMA
from lunarops.classes.time import Epoch
from lunarops.config.schema import ConfigSchema, path, sequence, string

from .table import AdditiveRangeBiasTable, RangeBiasLookup, RangeBiasLookupStatus


def _validate_range_bias_table(config: dict, path_name: str):
    selected = {key for key in ("file", "biases") if config.get(key) is not None}
    if len(selected) != 1:
        raise ValueError(f"{path_name} requires exactly one of 'file' or 'biases'.")
    if selected == {"file"}:
        incompatible = sorted(set(config) & {"source", "biases"})
        if incompatible:
            raise ValueError(f"{path_name} file form cannot include {incompatible}.")
    elif "file" in config:
        raise ValueError(f"{path_name} inline form cannot include 'file'.")
    return config


@dataclass(frozen=True, slots=True)
class RangeBiasRequest:
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
    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(fields=(), type_name="none")

    @classmethod
    def from_config(cls, cfg, ctx):
        return cls()

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
    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            path("file", non_empty=True),
            string("source", non_empty=True),
            sequence("biases", item_kind="mapping", item_nested=RANGE_BIAS_ROW_SCHEMA, min_items=1),
        ),
        type_name="table",
        validator=_validate_range_bias_table,
    )

    @classmethod
    def from_config(cls, cfg, ctx):
        from lunarops.classes.range_bias.table import (
            AdditiveRangeBiasTable,
            load_additive_range_bias_table,
        )

        if "file" in cfg:
            table = load_additive_range_bias_table(ctx.resolve_path(cfg["file"]))
        else:
            table = AdditiveRangeBiasTable.from_mapping(cfg)
        return cls(table)

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


class Inpop21aRangeBiasModel(TableRangeBiasModel):
    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(fields=(), type_name="inpop21a")

    @classmethod
    def from_config(cls, cfg, ctx):
        from .table import builtin_additive_range_bias_table

        return cls(builtin_additive_range_bias_table("inpop21a"))
