"""Configuration-backed additive station range-bias tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from lunarops.classes.time import Epoch
from lunarops.config.schema import ConfigSchema, number, sequence, string
from lunarops.config.schema import field as schema_field


class RangeBiasLookupStatus(StrEnum):
    """Diagnostic result of a range-bias table lookup."""

    MATCHED = "matched"
    EXPLICIT_ZERO = "explicit_zero"
    STATION_NOT_IN_TABLE = "station_not_in_table"
    OUTSIDE_COVERAGE = "outside_coverage"


RANGE_BIAS_ROW_SCHEMA = ConfigSchema(
    fields=(
        string("station", required=True, non_empty=True, allow_none=False),
        schema_field("start", "time", required=True, allow_none=False),
        schema_field("end", "time", required=True, allow_none=False),
        number("correctionTwoWayCm", required=True, allow_none=False),
        string("source", non_empty=True),
    )
)
RANGE_BIAS_TABLE_FILE_SCHEMA = ConfigSchema(
    fields=(
        string("type", non_empty=True),
        string("source", non_empty=True),
        sequence("biases", required=True, item_kind="mapping", item_nested=RANGE_BIAS_ROW_SCHEMA, min_items=1),
    )
)


def _parse_date(value: object, *, name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date | str):
        raise TypeError(f"{name} must be an ISO date string or datetime.date.")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be written as YYYY-MM-DD, got {value!r}.") from exc


@dataclass(frozen=True, slots=True)
class RangeBiasComponent:
    """One additive two-way range-bias component over a UTC date interval."""

    station_id: str
    start_date_utc: date
    end_date_exclusive_utc: date
    correction_two_way_cm: float
    source: str | None = None

    def __post_init__(self) -> None:
        if self.end_date_exclusive_utc <= self.start_date_utc:
            raise ValueError("end_date_exclusive_utc must be after start_date_utc.")

    def active_on(self, observation_date_utc: date) -> bool:
        return self.start_date_utc <= observation_date_utc < self.end_date_exclusive_utc


@dataclass(frozen=True, slots=True)
class RangeBiasLookup:
    """The components and status selected for one observation."""

    requested_station_id: str
    matched_station_id: str | None
    observation_date_utc: date
    active_components: tuple[RangeBiasComponent, ...]
    status: RangeBiasLookupStatus

    @property
    def correction_two_way_cm(self) -> float:
        return float(sum(component.correction_two_way_cm for component in self.active_components))

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(component.source for component in self.active_components if component.source is not None)
        )


def _builtin_date(value: str) -> date:
    return date.fromisoformat(value.replace("/", "-"))


# INPOP21a Table 8. Overlapping rows are intentionally additive components.
INPOP21A_RANGE_BIAS_COMPONENTS: tuple[RangeBiasComponent, ...] = (
    RangeBiasComponent("APOLLO", _builtin_date("2006/04/07"), _builtin_date("2010/11/01"), 0.03, "INPOP21a Table 8"),
    RangeBiasComponent("APOLLO", _builtin_date("2007/12/15"), _builtin_date("2008/06/30"), -3.93, "INPOP21a Table 8"),
    RangeBiasComponent("APOLLO", _builtin_date("2008/09/20"), _builtin_date("2009/06/20"), 3.22, "INPOP21a Table 8"),
    RangeBiasComponent("APOLLO", _builtin_date("2010/11/01"), _builtin_date("2012/04/07"), -6.28, "INPOP21a Table 8"),
    RangeBiasComponent("APOLLO", _builtin_date("2012/04/07"), _builtin_date("2013/09/02"), 8.85, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("1984/06/01"), _builtin_date("1986/06/13"), -17.12, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("1987/10/01"), _builtin_date("2005/08/01"), -5.41, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("1993/03/01"), _builtin_date("1996/10/01"), 9.81, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("1996/12/10"), _builtin_date("1997/01/18"), 14.32, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("1997/02/08"), _builtin_date("1998/06/24"), 20.79, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("2004/12/04"), _builtin_date("2004/12/07"), -5.53, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("2005/01/03"), _builtin_date("2005/01/06"), -4.53, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("2009/11/01"), _builtin_date("2014/01/01"), -0.99, "INPOP21a Table 8"),
    RangeBiasComponent("GRASSE", _builtin_date("2015/12/20"), _builtin_date("2015/12/21"), -88.05, "INPOP21a Table 8"),
    RangeBiasComponent(
        "HALEAKALA", _builtin_date("1984/11/01"), _builtin_date("1990/09/01"), 10.07, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        "HALEAKALA", _builtin_date("1984/11/01"), _builtin_date("1986/04/01"), -0.72, "INPOP21a Table 8"
    ),
    RangeBiasComponent("HALEAKALA", _builtin_date("1986/04/02"), _builtin_date("1987/07/30"), 9.81, "INPOP21a Table 8"),
    RangeBiasComponent("HALEAKALA", _builtin_date("1987/07/31"), _builtin_date("1987/08/14"), 1.86, "INPOP21a Table 8"),
    RangeBiasComponent(
        "HALEAKALA", _builtin_date("1985/06/09"), _builtin_date("1985/06/10"), -11.18, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        "HALEAKALA", _builtin_date("1987/11/10"), _builtin_date("1988/02/18"), 18.57, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        "HALEAKALA", _builtin_date("1990/02/06"), _builtin_date("1990/09/01"), 13.36, "INPOP21a Table 8"
    ),
    RangeBiasComponent("MATERA", _builtin_date("2003/01/01"), _builtin_date("2016/01/01"), 0.34, "INPOP21a Table 8"),
    RangeBiasComponent(
        "MCDONALD", _builtin_date("1969/01/01"), _builtin_date("1985/07/01"), -46.56, "INPOP21a Table 8"
    ),
    RangeBiasComponent("MCDONALD", _builtin_date("1971/12/01"), _builtin_date("1972/12/05"), 40.23, "INPOP21a Table 8"),
    RangeBiasComponent(
        "MCDONALD", _builtin_date("1972/04/21"), _builtin_date("1972/04/27"), 129.56, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        "MCDONALD", _builtin_date("1974/08/18"), _builtin_date("1974/10/16"), -114.07, "INPOP21a Table 8"
    ),
    RangeBiasComponent("MCDONALD", _builtin_date("1975/10/05"), _builtin_date("1976/03/01"), 26.87, "INPOP21a Table 8"),
    RangeBiasComponent(
        "MCDONALD", _builtin_date("1983/12/01"), _builtin_date("1984/01/17"), -12.80, "INPOP21a Table 8"
    ),
    RangeBiasComponent("MLRS1", _builtin_date("1983/08/01"), _builtin_date("1988/01/28"), 14.42, "INPOP21a Table 8"),
    RangeBiasComponent("WETTZELL", _builtin_date("2018/01/01"), _builtin_date("2025/01/01"), 0.0, "INPOP21a Table 8"),
)


@dataclass(frozen=True, slots=True)
class AdditiveRangeBiasTable:
    """Station-indexed table whose overlapping components are summed."""

    components: tuple[RangeBiasComponent, ...]
    source: str | None = None
    units: ClassVar[str] = "cm two-way light distance"
    _components_by_station: Mapping[str, tuple[RangeBiasComponent, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if len(set(components)) != len(components):
            raise ValueError("Range-bias components must not contain exact duplicates.")
        grouped: dict[str, list[RangeBiasComponent]] = {}
        for component in components:
            grouped.setdefault(component.station_id, []).append(component)
        by_station = {
            station_id: tuple(sorted(items, key=lambda item: (item.start_date_utc, item.end_date_exclusive_utc)))
            for station_id, items in grouped.items()
        }
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "source", None if self.source is None else self.source.strip() or None)
        object.__setattr__(self, "_components_by_station", MappingProxyType(by_station))

    def lookup(self, station_id: str, observation_epoch_utc: Epoch) -> RangeBiasLookup:
        observation_date = date.fromisoformat(observation_epoch_utc.date_iso())
        components = self._components_by_station.get(station_id)
        if components is None:
            return RangeBiasLookup(
                requested_station_id=station_id,
                matched_station_id=None,
                observation_date_utc=observation_date,
                active_components=(),
                status=RangeBiasLookupStatus.STATION_NOT_IN_TABLE,
            )
        active = tuple(component for component in components if component.active_on(observation_date))
        status = (
            RangeBiasLookupStatus.OUTSIDE_COVERAGE
            if not active
            else RangeBiasLookupStatus.EXPLICIT_ZERO
            if all(component.correction_two_way_cm == 0.0 for component in active)
            else RangeBiasLookupStatus.MATCHED
        )
        return RangeBiasLookup(
            requested_station_id=station_id,
            matched_station_id=station_id,
            observation_date_utc=observation_date,
            active_components=active,
            status=status,
        )

    def coverage_intervals_by_station(self) -> dict[str, tuple[tuple[str, str], ...]]:
        return {
            station_id: tuple(
                (component.start_date_utc.isoformat(), component.end_date_exclusive_utc.isoformat())
                for component in components
            )
            for station_id, components in self._components_by_station.items()
        }

    @classmethod
    def from_mapping(
        cls,
        config_mapping: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
    ) -> AdditiveRangeBiasTable:
        """Build a table from a mapping already resolved by a range-bias schema."""
        fallback_source = str(source_path) if source_path is not None else None
        raw_source = config_mapping.get("source")
        source = fallback_source if raw_source is None else raw_source.strip() or fallback_source
        components = tuple(
            RangeBiasComponent(
                station_id=row["station"],
                start_date_utc=_parse_date(row["start"], name="Range-bias row start"),
                end_date_exclusive_utc=_parse_date(row["end"], name="Range-bias row end"),
                correction_two_way_cm=float(row["correctionTwoWayCm"]),
                source=source if row.get("source") is None else row["source"].strip() or None,
            )
            for row in config_mapping["biases"]
        )
        return cls(components=components, source=source)


INPOP21A_RANGE_BIAS_TABLE = AdditiveRangeBiasTable(
    components=INPOP21A_RANGE_BIAS_COMPONENTS,
    source="INPOP21a Table 8",
)

BUILTIN_ADDITIVE_RANGE_BIAS_TABLES: Mapping[str, AdditiveRangeBiasTable] = MappingProxyType(
    {"inpop21a": INPOP21A_RANGE_BIAS_TABLE}
)


def builtin_additive_range_bias_table(name: str) -> AdditiveRangeBiasTable:
    return BUILTIN_ADDITIVE_RANGE_BIAS_TABLES[name.strip().lower()]


def load_additive_range_bias_table(path: str | Path) -> AdditiveRangeBiasTable:
    file = Path(path).expanduser()
    if file.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError(f"Range-bias tables must use .yml or .yaml: {file}")
    import yaml

    data = RANGE_BIAS_TABLE_FILE_SCHEMA.resolve(
        yaml.safe_load(file.read_text(encoding="utf-8")),
        path=f"range-bias table {file}",
    )
    return AdditiveRangeBiasTable.from_mapping(data, source_path=file)


__all__ = [
    "BUILTIN_ADDITIVE_RANGE_BIAS_TABLES",
    "INPOP21A_RANGE_BIAS_COMPONENTS",
    "INPOP21A_RANGE_BIAS_TABLE",
    "RANGE_BIAS_ROW_SCHEMA",
    "RANGE_BIAS_TABLE_FILE_SCHEMA",
    "AdditiveRangeBiasTable",
    "RangeBiasComponent",
    "RangeBiasLookup",
    "RangeBiasLookupStatus",
    "builtin_additive_range_bias_table",
    "load_additive_range_bias_table",
]
