import gzip
from pathlib import Path
from typing import Any, cast

import pytest

from lunarops.base.constants import C
from lunarops.config.context import RunContext
from lunarops.fileio.normal_points import (
    read_normal_points,
    write_normal_points,
)
from lunarops.fileio.formats.normal_point_sources import (
    read_normal_point_source,
)
from lunarops.fileio.yaml_artifact import read_structured_text
from lunarops.programs.normal_point_programs import normal_points_convert


def _write_crd(path: Path) -> None:
    path.write_text(
        "H1 CRD 2 2020 1 2 0\n"
        "H2 APOL 70610 1 1 0 APOL\n"
        "H3 APOLLO15 0 0 0 0\n"
        "H4 0 2020 1 2 0 0 0 2020 1 2 1 0 0 0 0 0 0\n"
        "C0 0 532.123 system\n"
        "20 100.123456789 900.123 280.123 50.25 0\n"
        "11 100.123456789 2.500000000123 system 2 300.5 42 12.3456 "
        "0 0 0 0 0 4.567\n",
        encoding="ascii",
    )


def _write_mini(path: Path) -> None:
    line = (
        f"{1:1d}{1:1d}{'20200102':8s}{'0001400000000':13s}"
        f"{25_000_000_000_000:14d}{3:1d}{'70610':5s}{42:3d}"
        f"{123:6d}{46:3d}{' ':1s}{90_012:6d}{70:4d}{50:2d}"
        f"{5_321:5d}{' ':1s}{300:4d}{'  ':2s}{'mini':9s}"
    )
    path.write_text(line + "\n", encoding="ascii")


def test_external_crd_import_preserves_precision(tmp_path):
    source = tmp_path / "sample.crd"
    _write_crd(source)

    dataset = read_normal_point_source(source)
    record = dataset.records[0]

    assert record.station_name == "APOLLO"
    assert record.station_code == "70610"
    assert record.reflector_name == "Apollo15"
    assert record.round_trip_time_s == pytest.approx(2.500000000123)
    assert record.uncertainty_two_way_s == pytest.approx(12.3456e-12)
    assert record.range_uncertainty_one_way_m == pytest.approx(0.5 * C * 12.3456e-12)
    assert record.pressure_hpa == pytest.approx(900.123)
    assert record.temperature_k == pytest.approx(280.123)


def test_crd_epoch_event_is_ignored_for_llr_transmit_epoch(tmp_path):
    source = tmp_path / "sample.crd"
    _write_crd(source)
    text = source.read_text(encoding="ascii").replace("system 2 300.5", "system 1 300.5")
    source.write_text(text, encoding="ascii")

    record = read_normal_point_source(source).records[0]

    assert record.transmit_epoch.isot(precision=6) == "2020-01-02T00:01:40.123457"


def test_invalid_crd_record_is_reported(tmp_path):
    source = tmp_path / "sample.crd"
    _write_crd(source)
    with source.open("a", encoding="ascii") as stream:
        stream.write("11 invalid invalid system 2 300.5 42 invalid\n")

    dataset = read_normal_point_source(source)

    assert len(dataset.records) == 1
    assert dataset.n_input_records == 2
    assert dataset.n_invalid_records == 1
    assert dataset.import_issues[0]["line"] == 8


def test_external_mini_import_is_available_only_at_converter_boundary(tmp_path):
    source = tmp_path / "sample.mini"
    _write_mini(source)

    with pytest.raises(ValueError, match=r"\.txt"):
        read_normal_points(source)
    dataset = read_normal_point_source(source)

    assert dataset.records[0].station_name == "APOLLO"
    assert dataset.records[0].round_trip_time_s == pytest.approx(2.5)
    assert dataset.records[0].uncertainty_two_way_s == pytest.approx(12.3e-12)


@pytest.mark.parametrize("name", ["canonical.txt", "canonical.txt.gz"])
def test_native_text_round_trip_preserves_values(tmp_path, name):
    source = tmp_path / "sample.crd"
    target = tmp_path / name
    _write_crd(source)
    original = read_normal_point_source(source)

    assert write_normal_points(original, target) == target
    recovered = read_normal_points(target)
    dispatched = read_normal_points(target)

    assert recovered.name == original.name
    assert recovered.n_input_records == original.n_input_records
    assert recovered.records[0].transmit_epoch == original.records[0].transmit_epoch
    assert recovered.records[0].round_trip_time_s == original.records[0].round_trip_time_s
    assert recovered.records[0].uncertainty_two_way_s == original.records[0].uncertainty_two_way_s
    assert dispatched.records[0].station_code == "70610"
    assert dispatched.records[0].reflector_name == "Apollo15"
    if name.endswith(".gz"):
        with gzip.open(target, "rt", encoding="utf-8") as stream:
            text = stream.read()
    else:
        text = target.read_text(encoding="utf-8")
    assert "Apollo15" in text
    assert "Apollo%20" not in text


def test_normal_points_convert_is_repeatable_inside_input_directory(tmp_path):
    source = tmp_path / "sample.crd"
    target = tmp_path / "canonical.txt.gz"
    _write_crd(source)
    context = RunContext(working_dir=tmp_path)
    config = {
        "inputFilesNormalPoints": ["."],
        "datasetName": "campaign",
        "outputFileNormalPoints": "canonical.txt.gz",
        "outputFileImportReport": "importReport.txt.gz",
    }

    assert normal_points_convert(config, context) == target
    assert normal_points_convert(config, context) == target

    recovered = read_normal_points(target)
    report = read_structured_text(tmp_path / "importReport.txt.gz", "normalPointImportReport")
    assert recovered.name == "campaign"
    assert len(recovered.records) == 1
    assert report["recordCount"] == 1
    assert report["invalidRecordCount"] == 0


def test_mini_import_issues_are_published_without_an_implicit_log(tmp_path):
    source = tmp_path / "sample.mini"
    _write_mini(source)
    with source.open("a", encoding="ascii") as stream:
        stream.write("invalid record\n")
    context = RunContext(working_dir=tmp_path)

    normal_points_convert(
        {
            "inputFilesNormalPoints": ["sample.mini"],
            "outputFileNormalPoints": "normalPoints.txt.gz",
            "outputFileImportReport": "importReport.txt.gz",
        },
        context,
    )
    report = read_structured_text(tmp_path / "importReport.txt.gz", "normalPointImportReport")

    assert report["invalidRecordCount"] == 1
    report_sources = cast(list[dict[str, Any]], report["sources"])
    assert report_sources[0]["issues"][0]["line"] == 2
    assert not (tmp_path / "llr_mini_io_warnings.log").exists()


def test_native_writer_rejects_old_or_untyped_extensions(tmp_path):
    source = tmp_path / "sample.crd"
    _write_crd(source)
    dataset = read_normal_point_source(source)

    for name in ("normalPoints.jsonl", "normalPoints.csv", "normalPoints.llnpt"):
        with pytest.raises(ValueError, match=r"\.txt"):
            write_normal_points(dataset, tmp_path / name)
