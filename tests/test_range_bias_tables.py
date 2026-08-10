from datetime import date
from pathlib import Path

import pytest

from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.observation_factory import ensure_registered
from lunarops.classes.range_bias.models import RangeBiasRequest, TableRangeBiasModel
from lunarops.classes.range_bias.table import (
    BUILTIN_ADDITIVE_RANGE_BIAS_TABLES,
    INPOP21A_RANGE_BIAS_TABLE,
    AdditiveRangeBiasTable,
    RangeBiasComponent,
    RangeBiasLookupStatus,
    load_additive_range_bias_table,
)
from lunarops.config.registry import UnknownClassError, create


def _epoch(value: str) -> Epoch:
    return Epoch.from_isot(value, scale=TimeScale.UTC)


def test_inpop21a_is_the_explicit_name_for_the_builtin_table():
    assert BUILTIN_ADDITIVE_RANGE_BIAS_TABLES["inpop21a"] is INPOP21A_RANGE_BIAS_TABLE


def test_builtin_range_bias_lookup_uses_aliases_and_reports_coverage():
    epoch = _epoch("2008-01-01T00:00:00")
    lookup_by_code = INPOP21A_RANGE_BIAS_TABLE.lookup(("7045",), epoch)
    lookup_by_name = INPOP21A_RANGE_BIAS_TABLE.lookup(("APOLLO",), epoch)

    assert lookup_by_code.active_components == lookup_by_name.active_components
    assert lookup_by_code.matched_station_id == lookup_by_name.matched_station_id
    assert lookup_by_name.status is RangeBiasLookupStatus.MATCHED
    assert lookup_by_name.matched_station_id == "APOLLO"
    assert lookup_by_name.correction_two_way_cm == pytest.approx(-3.90)
    assert "APOLLO" in INPOP21A_RANGE_BIAS_TABLE.coverage_intervals_by_station()


def test_station_candidates_fall_through_unknown_catalog_identifier():
    epoch = _epoch("2010-06-01T00:00:00")
    lookup = INPOP21A_RANGE_BIAS_TABLE.lookup(("custom_catalog_key", "Grasse", "01910"), epoch)

    assert lookup.status is RangeBiasLookupStatus.MATCHED
    assert lookup.matched_station_id == "GRASSE"
    assert lookup.correction_two_way_cm == pytest.approx(-0.99)


def test_table_model_uses_ordered_station_identifiers():
    table_model = TableRangeBiasModel(INPOP21A_RANGE_BIAS_TABLE)
    correction = table_model.evaluate(RangeBiasRequest(("APOLLO",), _epoch("2008-01-01T00:00:00")))

    assert correction.correction_two_way_cm == pytest.approx(-3.90)
    assert correction.lookup.status is RangeBiasLookupStatus.MATCHED


def test_table_lookup_distinguishes_unresolved_unknown_and_outside_coverage():
    epoch = _epoch("2020-06-01T00:00:00")

    unknown = INPOP21A_RANGE_BIAS_TABLE.lookup(("not-a-station",), epoch)
    outside = INPOP21A_RANGE_BIAS_TABLE.lookup(("APOLLO",), epoch)
    explicit_zero = INPOP21A_RANGE_BIAS_TABLE.lookup(("WETTZELL",), epoch)

    assert unknown.status is RangeBiasLookupStatus.STATION_NOT_IN_TABLE
    assert outside.status is RangeBiasLookupStatus.OUTSIDE_COVERAGE
    assert explicit_zero.status is RangeBiasLookupStatus.EXPLICIT_ZERO
    assert explicit_zero.correction_two_way_cm == 0.0


def test_overlapping_components_are_additive():
    epoch = _epoch("2008-01-01T00:00:00")
    active = INPOP21A_RANGE_BIAS_TABLE.active_components(("APOLLO",), epoch)

    assert len(active) == 2
    assert INPOP21A_RANGE_BIAS_TABLE.total_correction_two_way_cm(("APOLLO",), epoch) == pytest.approx(-3.90)


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

    assert isinstance(table, AdditiveRangeBiasTable)
    assert table.source == str(path)
    assert table.total_correction_two_way_cm(("APOLLO",), epoch) == 12.5
    assert table.total_correction_two_way_cm(("GRASSE",), epoch) == -0.5


def test_range_bias_table_rejects_json_filename(tmp_path: Path):
    path = tmp_path / "range_bias.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.yml"):
        load_additive_range_bias_table(path)


def test_declarative_range_bias_rejects_old_entries_key():
    with pytest.raises(ValueError, match="unknown key"):
        AdditiveRangeBiasTable.from_mapping(
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
        with pytest.raises(ValueError, match="unknown key"):
            AdditiveRangeBiasTable.from_mapping(config)


def test_declarative_range_bias_canonicalizes_station_aliases_on_load():
    epoch = _epoch("2020-06-01T00:00:00")
    table = AdditiveRangeBiasTable.from_mapping(
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

    assert table.total_correction_two_way_cm(("APOL", "70610"), epoch) == 7.0


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
        AdditiveRangeBiasTable.from_mapping({"biases": [row]})


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
        AdditiveRangeBiasTable.from_mapping({"biases": [row]})


def test_range_bias_component_rejects_datetime_and_duplicate_components():
    component = RangeBiasComponent("APOLLO", date(2020, 1, 1), date(2021, 1, 1), 1.0)
    with pytest.raises(ValueError, match="exact duplicates"):
        AdditiveRangeBiasTable((component, component))


def test_builtin_registry_is_read_only():
    with pytest.raises(TypeError):
        BUILTIN_ADDITIVE_RANGE_BIAS_TABLES["custom"] = INPOP21A_RANGE_BIAS_TABLE  # type: ignore[index]


def test_factory_exposes_only_inpop21a():
    ensure_registered()

    with pytest.raises(UnknownClassError):
        create("rangeBias", "inpop21")
    model = create("rangeBias", "inpop21a")

    assert isinstance(model, TableRangeBiasModel)


def test_factory_file_config_rejects_unknown_keys(tmp_path: Path):
    ensure_registered()
    path = tmp_path / "range_bias.yml"
    path.write_text(
        "biases:\n  - {station: APOLLO, start: 2020-01-01, end: 2021-01-01, correctionTwoWayCm: 1.0}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="file form cannot include"):
        create("rangeBias", {"type": "table", "file": str(path), "source": "unexpected"})
