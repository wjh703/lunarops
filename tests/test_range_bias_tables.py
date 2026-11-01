from datetime import date
from pathlib import Path

import pytest

from lunarops.classes.range_bias.table import (
    BUILTIN_ADDITIVE_RANGE_BIAS_TABLES,
    INPOP21A_RANGE_BIAS_TABLE,
    RangeBiasComponent,
    RangeBiasTable,
    load_additive_range_bias_table,
)
from lunarops.config.registry import create, resolve


def _create(category, config):
    return create(category, resolve(category, config))


def _epoch(value: str) -> date:
    return date.fromisoformat(value[:10])


def test_inpop21a_is_the_explicit_name_for_the_builtin_table():
    assert BUILTIN_ADDITIVE_RANGE_BIAS_TABLES["inpop21a"] is INPOP21A_RANGE_BIAS_TABLE


def test_builtin_range_bias_lookup_requires_exact_id_and_reports_coverage():
    table = INPOP21A_RANGE_BIAS_TABLE
    day = _epoch("2008-01-01")
    code = table.lookup("7045", day)
    name = table.lookup("APOLLO", day)
    assert table.diagnostics("7045", *code)["range_bias_lookup_status"] == "station_not_in_table"
    assert table.diagnostics("APOLLO", *name)["range_bias_lookup_status"] == "matched"
    assert table.diagnostics("APOLLO", *name)["range_bias_station_id"] == "APOLLO"
    assert name[0] == pytest.approx(-3.90)


def test_empty_table_is_the_zero_model():
    table = RangeBiasTable()
    value, components = table.lookup("APOLLO", _epoch("2008-01-01"))
    assert value == 0.0 and components == ()
    diagnostics = table.diagnostics("APOLLO", value, components)
    assert diagnostics["range_bias_model_label"] == "none"
    assert diagnostics["range_bias_lookup_status"] == "explicit_zero"


def test_table_lookup_distinguishes_unknown_outside_and_explicit_zero():
    table = INPOP21A_RANGE_BIAS_TABLE
    day = _epoch("2020-06-01")
    assert table.diagnostics("not-a-station", *table.lookup("not-a-station", day))["range_bias_lookup_status"] == "station_not_in_table"
    assert table.diagnostics("APOLLO", *table.lookup("APOLLO", day))["range_bias_lookup_status"] == "outside_coverage"
    zero = table.lookup("WETTZELL", day)
    assert table.diagnostics("WETTZELL", *zero)["range_bias_lookup_status"] == "explicit_zero"
    assert zero[0] == 0.0


def test_overlapping_components_are_additive():
    value, components = INPOP21A_RANGE_BIAS_TABLE.lookup("APOLLO", _epoch("2008-01-01"))
    assert len(components) == 2
    assert value == pytest.approx(-3.90)


def test_declarative_range_bias_table_from_canonical_yaml_rows(tmp_path: Path):
    path = tmp_path / "range_bias.yml"
    path.write_text(
        """
biases:
  - {station: APOLLO, start: 2020-01-01, end: 2021-01-01, correctionTwoWayCm: 12.5}
  - {station: GRASSE, start: 2020-01-01, end: 2021-01-01, correctionTwoWayCm: -0.5}
""".strip(),
        encoding="utf-8",
    )
    table = load_additive_range_bias_table(path)
    epoch = _epoch("2020-06-01T00:00:00")

    assert isinstance(table, RangeBiasTable)
    assert table.source == str(path)
    assert table.lookup("APOLLO", epoch)[0] == 12.5
    assert table.lookup("GRASSE", epoch)[0] == -0.5


def test_range_bias_table_rejects_json_filename(tmp_path: Path):
    path = tmp_path / "range_bias.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.yml"):
        load_additive_range_bias_table(path)


def _create_inline_table(config):
    return _create("rangeBias", {"type": "table", **config})


def test_declarative_range_bias_rejects_old_entries_key():
    with pytest.raises(ValueError, match="unknown configuration key"):
        _create_inline_table(
            {
                "entries": [
                    {
                        "station": "APOLLO",
                        "start": "2020-01-01",
                        "end": "2021-01-01",
                        "correctionTwoWayCm": 1.0,
                    }
                ],
            }
        )


def test_declarative_range_bias_rejects_old_bias_key_and_alias_config():
    for config in (
        {
            "name": "custom",
            "biases": [
                {
                    "station": "APOLLO",
                    "start": "2020-01-01",
                    "end": "2021-01-01",
                    "correctionTwoWayCm": 1.0,
                }
            ],
        },
        {
            "biases": [
                {
                    "station": "APOLLO",
                    "start": "2020-01-01",
                    "end": "2021-01-01",
                    "biasCm": 1.0,
                }
            ],
        },
    ):
        with pytest.raises(ValueError, match="unknown configuration key"):
            _create_inline_table(config)


def test_declarative_range_bias_preserves_exact_custom_id():
    epoch = _epoch("2020-06-01T00:00:00")
    table = RangeBiasTable.from_mapping(
        {
            "biases": [
                {
                    "station": "APOL",
                    "start": "2020-01-01",
                    "end": "2021-01-01",
                    "correctionTwoWayCm": 7.0,
                }
            ]
        }
    )

    assert table.lookup("APOL", epoch)[0] == 7.0


@pytest.mark.parametrize(
    "row",
    [
        "APOLLO 2020-01-01/2021-01-01 1.0",
        ["APOLLO", "2020-01-01", "2021-01-01", 1.0],
        {
            "station": "APOLLO",
            "interval": "2020-01-01/2021-01-01",
            "correctionTwoWayCm": 1.0,
        },
    ],
)
def test_range_bias_rejects_noncanonical_row_forms(row):
    with pytest.raises((TypeError, ValueError)):
        _create_inline_table({"biases": [row]})


@pytest.mark.parametrize("field,value", [("station", 7045), ("correctionTwoWayCm", True)])
def test_range_bias_rejects_implicitly_coerced_values(field, value):
    row = {
        "station": "APOLLO",
        "start": "2020-01-01",
        "end": "2021-01-01",
        "correctionTwoWayCm": 1.0,
    }
    row[field] = value

    with pytest.raises(TypeError):
        _create_inline_table({"biases": [row]})


def test_range_bias_component_rejects_datetime_and_duplicate_components():
    component = RangeBiasComponent("APOLLO", date(2020, 1, 1), date(2021, 1, 1), 1.0)
    with pytest.raises(ValueError, match="exact duplicates"):
        RangeBiasTable((component, component))


def test_builtin_registry_is_read_only():
    with pytest.raises(TypeError):
        BUILTIN_ADDITIVE_RANGE_BIAS_TABLES["custom"] = INPOP21A_RANGE_BIAS_TABLE  # type: ignore[index]


def test_factory_exposes_only_inpop21a():

    with pytest.raises(KeyError):
        _create("rangeBias", {"type": "inpop21"})
    model = _create("rangeBias", {"type": "inpop21a"})

    assert isinstance(model, RangeBiasTable)


def test_factory_file_config_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "range_bias.yml"
    path.write_text(
        "biases:\n  - {station: APOLLO, start: 2020-01-01, end: 2021-01-01, correctionTwoWayCm: 1.0}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="file form cannot include"):
        _create("rangeBias", {"type": "table", "file": str(path), "source": "unexpected"})
