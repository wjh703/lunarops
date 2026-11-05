"""Configuration-backed additive station range-bias tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from lunarops.base.constants import C
from lunarops.base.station_interval import StationInterval
from lunarops.config.schema import ConfigSchema, number, path, sequence, string
from lunarops.config.schema import field as schema_field


def _validate_range_bias_table(config: dict, path_name: str):
    if config["type"] != "table":
        unknown = set(config) - {"type"}
        if unknown:
            raise ValueError(f"{path_name} unknown configuration key(s): {sorted(unknown)}")
        return config
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


RANGE_BIAS_ROW_SCHEMA = ConfigSchema(
    fields=(
        string("station", required=True, non_empty=True, description="Station catalog key."),
        schema_field(
            "start",
            "time",
            required=True,
            description="First UTC date of the correction (inclusive).",
        ),
        schema_field(
            "end",
            "time",
            required=True,
            description="UTC date at which the correction ends (exclusive).",
        ),
        number(
            "correctionTwoWayCm",
            required=True,
            description=(
                "Correction in centimetres of two-way light distance; it is subtracted from the computed round-trip "
                "time. Overlapping rows add up."
            ),
        ),
        string("source", non_empty=True, description="Provenance of the row."),
    )
)
RANGE_BIAS_TABLE_FILE_SCHEMA = ConfigSchema(
    fields=(
        string("type", non_empty=True, description="Table label stored in the file."),
        string("source", non_empty=True, description="Provenance of the table."),
        sequence(
            "biases",
            required=True,
            item_kind="mapping",
            item_nested=RANGE_BIAS_ROW_SCHEMA,
            min_items=1,
            description="Correction rows.",
        ),
    )
)


@dataclass(frozen=True, slots=True)
class RangeBiasComponent:
    """One additive two-way range-bias component of a station over a UTC date interval.

    Attributes
    ----------
    interval : StationInterval
        Station and UTC date interval (start inclusive, end exclusive).
    correction_two_way_cm : float
        Correction in centimetres of two-way light distance; it is subtracted from the computed
        round-trip time.
    source : str or None
        Provenance of the component.
    """

    interval: StationInterval
    correction_two_way_cm: float
    source: str | None = None


def _builtin_date(value: str) -> date:
    return StationInterval.parse(value.replace("/", "-"))


# INPOP21a Table 8. Overlapping rows are intentionally additive components.
INPOP21A_RANGE_BIAS_COMPONENTS: tuple[RangeBiasComponent, ...] = (
    RangeBiasComponent(
        StationInterval("APOLLO", _builtin_date("2006/04/07"), _builtin_date("2010/11/01")), 0.03, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("APOLLO", _builtin_date("2007/12/15"), _builtin_date("2008/06/30")), -3.93, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("APOLLO", _builtin_date("2008/09/20"), _builtin_date("2009/06/20")), 3.22, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("APOLLO", _builtin_date("2010/11/01"), _builtin_date("2012/04/07")), -6.28, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("APOLLO", _builtin_date("2012/04/07"), _builtin_date("2013/09/02")), 8.85, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("1984/06/01"), _builtin_date("1986/06/13")), -17.12, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("1987/10/01"), _builtin_date("2005/08/01")), -5.41, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("1993/03/01"), _builtin_date("1996/10/01")), 9.81, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("1996/12/10"), _builtin_date("1997/01/18")), 14.32, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("1997/02/08"), _builtin_date("1998/06/24")), 20.79, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("2004/12/04"), _builtin_date("2004/12/07")), -5.53, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("2005/01/03"), _builtin_date("2005/01/06")), -4.53, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("2009/11/01"), _builtin_date("2014/01/01")), -0.99, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("GRASSE", _builtin_date("2015/12/20"), _builtin_date("2015/12/21")), -88.05, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("HALEAKALA", _builtin_date("1984/11/01"), _builtin_date("1990/09/01")),
        10.07,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("HALEAKALA", _builtin_date("1984/11/01"), _builtin_date("1986/04/01")),
        -0.72,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("HALEAKALA", _builtin_date("1986/04/02"), _builtin_date("1987/07/30")), 9.81, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("HALEAKALA", _builtin_date("1987/07/31"), _builtin_date("1987/08/14")), 1.86, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("HALEAKALA", _builtin_date("1985/06/09"), _builtin_date("1985/06/10")),
        -11.18,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("HALEAKALA", _builtin_date("1987/11/10"), _builtin_date("1988/02/18")),
        18.57,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("HALEAKALA", _builtin_date("1990/02/06"), _builtin_date("1990/09/01")),
        13.36,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("MATERA", _builtin_date("2003/01/01"), _builtin_date("2016/01/01")), 0.34, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("MCDONALD", _builtin_date("1969/01/01"), _builtin_date("1985/07/01")),
        -46.56,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("MCDONALD", _builtin_date("1971/12/01"), _builtin_date("1972/12/05")), 40.23, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("MCDONALD", _builtin_date("1972/04/21"), _builtin_date("1972/04/27")),
        129.56,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("MCDONALD", _builtin_date("1974/08/18"), _builtin_date("1974/10/16")),
        -114.07,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("MCDONALD", _builtin_date("1975/10/05"), _builtin_date("1976/03/01")), 26.87, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("MCDONALD", _builtin_date("1983/12/01"), _builtin_date("1984/01/17")),
        -12.80,
        "INPOP21a Table 8",
    ),
    RangeBiasComponent(
        StationInterval("MLRS1", _builtin_date("1983/08/01"), _builtin_date("1988/01/28")), 14.42, "INPOP21a Table 8"
    ),
    RangeBiasComponent(
        StationInterval("WETTZELL", _builtin_date("2018/01/01"), _builtin_date("2025/01/01")), 0.0, "INPOP21a Table 8"
    ),
)


@dataclass(frozen=True, slots=True)
class RangeBiasTable:
    """Additive station range-bias table of two-way corrections over UTC date intervals.

    The configuration ``type`` selects the table: ``none`` (empty), ``inpop21a`` (the built-in INPOP21a
    Table 8 components) or ``table`` (rows from ``file`` or inline ``biases``). Components of the same
    station whose intervals contain the observation date are summed. Corrections are in centimetres of
    two-way light distance and are subtracted from the computed round-trip time.

    Parameters
    ----------
    components : tuple of RangeBiasComponent, optional
        Table rows; exact duplicates are rejected.
    source : str or None, optional
        Provenance of the table; blank text is stored as ``None``.

    Attributes
    ----------
    units : str
        Unit of the corrections, ``"cm two-way light distance"``.

    Raises
    ------
    ValueError
        If ``components`` contains exact duplicates.

    References
    ----------
    .. [1] Fienga, A., Deram, P., Di Ruscio, A., et al. (2021). INPOP21a planetary ephemerides. Notes
       Scientifiques et Techniques de l'Institut de Mécanique Céleste, 110, Table 8.
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            path(
                "file",
                non_empty=True,
                description="YAML range-bias table (`.yml`/`.yaml`) with `biases` rows. Only for `type: table`.",
            ),
            string("source", non_empty=True, description="Provenance of the inline table. Only for `type: table`."),
            sequence(
                "biases",
                item_kind="mapping",
                item_nested=RANGE_BIAS_ROW_SCHEMA,
                min_items=1,
                description="Inline correction rows. Only for `type: table`; give either this or `file`.",
            ),
        ),
        validator=_validate_range_bias_table,
    )

    @classmethod
    def from_config(cls, cfg, ctx):
        """Create the table selected by the configuration ``type`` (``none``, ``inpop21a`` or ``table``)."""
        if cfg["type"] == "none":
            return cls()
        if cfg["type"] == "inpop21a":
            return INPOP21A_RANGE_BIAS_TABLE
        if "file" in cfg:
            return load_additive_range_bias_table(ctx.resolve_path(cfg["file"]))
        return cls.from_mapping(cfg)

    components: tuple[RangeBiasComponent, ...] = ()
    source: str | None = None
    units: ClassVar[str] = "cm two-way light distance"
    _components_by_station: Mapping[str, tuple[RangeBiasComponent, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the components and index them by station."""
        components = tuple(self.components)
        if len(set(components)) != len(components):
            raise ValueError("Range-bias components must not contain exact duplicates.")
        grouped: dict[str, list[RangeBiasComponent]] = {}
        for component in components:
            grouped.setdefault(component.interval.station, []).append(component)
        by_station = {
            station_id: tuple(sorted(items, key=lambda item: (item.interval.start, item.interval.end_exclusive)))
            for station_id, items in grouped.items()
        }
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "source", None if self.source is None else self.source.strip() or None)
        object.__setattr__(self, "_components_by_station", MappingProxyType(by_station))

    @property
    def model_label(self) -> str:
        """Label of the table for diagnostics: its source, a generic label, or ``"none"`` if empty."""
        return (self.source or "additive range-bias table") if self.components else "none"

    def lookup(self, station: str, day: date) -> tuple[float, tuple[RangeBiasComponent, ...]]:
        """Return the total correction of a station on a UTC date and the components contributing to it.

        Parameters
        ----------
        station : str
            Station catalog key.
        day : date
            UTC calendar date of the observation.

        Returns
        -------
        correction_two_way_cm : float
            Sum of the active corrections, in centimetres of two-way light distance; 0 if none is active.
        components : tuple of RangeBiasComponent
            Active components, ordered by interval start and end.
        """
        active = tuple(
            component
            for component in self._components_by_station.get(station, ())
            if component.interval.contains(station, day)
        )
        return float(sum(component.correction_two_way_cm for component in active)), active

    def diagnostics(self, station: str, cm: float, components: tuple[RangeBiasComponent, ...]) -> dict[str, object]:
        """Return diagnostic fields describing a range-bias lookup.

        Parameters
        ----------
        station : str
            Station catalog key.
        cm : float
            Total correction from ``lookup``, in centimetres of two-way light distance.
        components : tuple of RangeBiasComponent
            Active components from ``lookup``.

        Returns
        -------
        dict
            ``range_bias_*`` fields: table label; lookup status (``matched``, ``explicit_zero``,
            ``station_not_in_table`` or ``outside_coverage``); station; number and sources of the active
            components; and the correction as two-way distance in cm and m, round-trip time in s
            (two-way distance / c) and one-way distance in m.
        """
        matched = station in self._components_by_station
        status = (
            "explicit_zero"
            if not self.components
            else "station_not_in_table"
            if not matched
            else "outside_coverage"
            if not components
            else "explicit_zero"
            if all(component.correction_two_way_cm == 0.0 for component in components)
            else "matched"
        )
        sources = tuple(dict.fromkeys(component.source for component in components if component.source is not None))
        two_way_m = 0.01 * cm
        return {
            "range_bias_model_label": self.model_label,
            "range_bias_lookup_status": status,
            "range_bias_station_id": station if matched else None,
            "range_bias_active_component_count": len(components),
            "range_bias_sources": "+".join(sources) or None,
            "range_bias_correction_two_way_cm": cm,
            "range_bias_correction_two_way_m": two_way_m,
            "range_bias_correction_round_trip_time_s": two_way_m / C,
            "range_bias_correction_one_way_m": 0.5 * two_way_m,
        }

    @classmethod
    def from_mapping(
        cls,
        config_mapping: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
    ) -> RangeBiasTable:
        """Create a table from a validated mapping with ``biases`` rows.

        Parameters
        ----------
        config_mapping : Mapping
            Mapping with optional ``source`` and a ``biases`` sequence of rows with ``station``,
            ``start``, ``end``, ``correctionTwoWayCm`` and optional ``source``.
        source_path : str or Path or None, optional
            File the mapping was read from; used as the source when the mapping has none.

        Returns
        -------
        RangeBiasTable
            Table whose rows without their own ``source`` inherit the table source.
        """
        fallback_source = str(source_path) if source_path is not None else None
        raw_source = config_mapping.get("source")
        source = fallback_source if raw_source is None else raw_source.strip() or fallback_source
        components = tuple(
            RangeBiasComponent(
                interval=StationInterval(
                    row["station"], StationInterval.parse(row["start"]), StationInterval.parse(row["end"])
                ),
                correction_two_way_cm=float(row["correctionTwoWayCm"]),
                source=source if row.get("source") is None else row["source"].strip() or None,
            )
            for row in config_mapping["biases"]
        )
        return cls(components=components, source=source)


INPOP21A_RANGE_BIAS_TABLE = RangeBiasTable(
    components=INPOP21A_RANGE_BIAS_COMPONENTS,
    source="INPOP21a Table 8",
)

BUILTIN_ADDITIVE_RANGE_BIAS_TABLES: Mapping[str, RangeBiasTable] = MappingProxyType(
    {"inpop21a": INPOP21A_RANGE_BIAS_TABLE}
)


def builtin_additive_range_bias_table(name: str) -> RangeBiasTable:
    """Return the built-in table with a case-insensitive name (``"inpop21a"``); raise ``KeyError`` otherwise."""
    return BUILTIN_ADDITIVE_RANGE_BIAS_TABLES[name.strip().lower()]


def load_additive_range_bias_table(path: str | Path) -> RangeBiasTable:
    """Read a range-bias table from a YAML file.

    Parameters
    ----------
    path : str or Path
        ``.yml`` or ``.yaml`` file with optional ``type`` and ``source`` and a ``biases`` list of rows.

    Returns
    -------
    RangeBiasTable
        Table whose source defaults to the file path.

    Raises
    ------
    ValueError
        If the file extension is not ``.yml``/``.yaml`` or the content does not match the table schema.
    """
    file = Path(path).expanduser()
    if file.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError(f"Range-bias tables must use .yml or .yaml: {file}")
    import yaml

    data = RANGE_BIAS_TABLE_FILE_SCHEMA.resolve(
        yaml.safe_load(file.read_text(encoding="utf-8")),
        path=f"range-bias table {file}",
    )
    return RangeBiasTable.from_mapping(data, source_path=file)


__all__ = [
    "BUILTIN_ADDITIVE_RANGE_BIAS_TABLES",
    "INPOP21A_RANGE_BIAS_COMPONENTS",
    "INPOP21A_RANGE_BIAS_TABLE",
    "RANGE_BIAS_ROW_SCHEMA",
    "RANGE_BIAS_TABLE_FILE_SCHEMA",
    "RangeBiasComponent",
    "RangeBiasTable",
    "builtin_additive_range_bias_table",
    "load_additive_range_bias_table",
]
