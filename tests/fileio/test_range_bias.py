"""Additive station range-bias tables: the INPOP21A table, YAML tables and the factory."""

from datetime import date

import pytest

from lunarops.base.station_interval import StationInterval
from lunarops.classes.range_bias.table import (
    BUILTIN_ADDITIVE_RANGE_BIAS_TABLES,
    INPOP21A_RANGE_BIAS_TABLE,
    RangeBiasComponent,
    RangeBiasTable,
    load_additive_range_bias_table,
)
from lunarops.config.registry import create, resolve

ROW = {"station": "APOLLO", "start": "2020-01-01", "end": "2021-01-01", "correctionTwoWayCm": 1.0}
JUNE_2020 = date(2020, 6, 1)


def _factory(config):
    return create("rangeBias", resolve("rangeBias", config))


def _status(table, station, day):
    return table.diagnostics(station, *table.lookup(station, day))["range_bias_lookup_status"]


def test_inpop21a_is_the_only_builtin_table_and_cannot_be_replaced():
    assert BUILTIN_ADDITIVE_RANGE_BIAS_TABLES["inpop21a"] is INPOP21A_RANGE_BIAS_TABLE
    assert isinstance(_factory({"type": "inpop21a"}), RangeBiasTable)
    with pytest.raises(KeyError):
        _factory({"type": "inpop21"})
    with pytest.raises(TypeError):
        BUILTIN_ADDITIVE_RANGE_BIAS_TABLES["custom"] = INPOP21A_RANGE_BIAS_TABLE  # type: ignore[index]


def test_inpop21a_apollo_2008_sums_its_two_overlapping_components():
    value, components = INPOP21A_RANGE_BIAS_TABLE.lookup("APOLLO", date(2008, 1, 1))
    diagnostics = INPOP21A_RANGE_BIAS_TABLE.diagnostics("APOLLO", value, components)

    assert len(components) == 2 and value == pytest.approx(-3.90)
    assert (diagnostics["range_bias_lookup_status"], diagnostics["range_bias_station_id"]) == ("matched", "APOLLO")
    assert _status(INPOP21A_RANGE_BIAS_TABLE, "7045", date(2008, 1, 1)) == "station_not_in_table"


def test_lookups_distinguish_unknown_stations_coverage_gaps_and_explicit_zeros():
    assert _status(INPOP21A_RANGE_BIAS_TABLE, "not-a-station", JUNE_2020) == "station_not_in_table"
    assert _status(INPOP21A_RANGE_BIAS_TABLE, "APOLLO", JUNE_2020) == "outside_coverage"
    assert INPOP21A_RANGE_BIAS_TABLE.lookup("WETTZELL", JUNE_2020)[0] == 0.0
    assert _status(INPOP21A_RANGE_BIAS_TABLE, "WETTZELL", JUNE_2020) == "explicit_zero"
    empty = RangeBiasTable()
    assert empty.lookup("APOLLO", date(2008, 1, 1)) == (0.0, ())
    assert empty.diagnostics("APOLLO", 0.0, ())["range_bias_model_label"] == "none"


def test_yaml_tables_load_canonical_rows_and_keep_exact_station_ids(tmp_path):
    path = tmp_path / "range_bias.yml"
    path.write_text("biases:\n"
                    "  - {station: APOLLO, start: 2020-01-01, end: 2021-01-01, correctionTwoWayCm: 12.5}\n"
                    "  - {station: APOL, start: 2020-01-01, end: 2021-01-01, correctionTwoWayCm: -0.5}\n")
    table = load_additive_range_bias_table(path)

    assert table.source == str(path)
    assert (table.lookup("APOLLO", JUNE_2020)[0], table.lookup("APOL", JUNE_2020)[0]) == (12.5, -0.5)
    (tmp_path / "range_bias.json").write_text("{}")
    with pytest.raises(ValueError, match=r"\.yml"):
        load_additive_range_bias_table(tmp_path / "range_bias.json")
    with pytest.raises(ValueError, match="file form cannot include"):
        _factory({"type": "table", "file": str(path), "source": "unexpected"})


@pytest.mark.parametrize("config", [
    {"entries": [ROW]},
    {"name": "custom", "biases": [ROW]},
    {"biases": [{**{k: v for k, v in ROW.items() if k != "correctionTwoWayCm"}, "biasCm": 1.0}]},
], ids=["entries-key", "name-key", "bias-cm-key"])
def test_inline_tables_reject_undeclared_keys(config):
    with pytest.raises(ValueError, match="unknown configuration key"):
        _factory({"type": "table", **config})


@pytest.mark.parametrize("row", [
    "APOLLO 2020-01-01/2021-01-01 1.0",
    ["APOLLO", "2020-01-01", "2021-01-01", 1.0],
    {"station": "APOLLO", "interval": "2020-01-01/2021-01-01", "correctionTwoWayCm": 1.0},
    {**ROW, "station": 7045},
    {**ROW, "correctionTwoWayCm": True},
], ids=["text-row", "list-row", "interval-row", "numeric-station", "boolean-correction"])
def test_inline_tables_reject_non_canonical_rows(row):
    with pytest.raises((TypeError, ValueError)):
        _factory({"type": "table", "biases": [row]})


def test_duplicate_components_are_rejected():
    component = RangeBiasComponent(StationInterval("APOLLO", date(2020, 1, 1), date(2021, 1, 1)), 1.0)
    with pytest.raises(ValueError, match="exact duplicates"):
        RangeBiasTable((component, component))
