"""CRD and MINI imports, native normal-point files, dataset filters and the converter program."""

import gzip
from dataclasses import replace

import pytest

from lunarops.base.constants import C
from lunarops.classes.observation.normal_points import NptDataset, combine_npt_datasets
from lunarops.config.context import RunContext
from lunarops.fileio.formats.crd import parse_crd_file
from lunarops.fileio.formats.mini import parse_mini_file
from lunarops.fileio.formats.normal_point_sources import read_normal_point_source
from lunarops.fileio.normal_points import read_normal_points, write_normal_points
from lunarops.fileio.yaml_artifact import read_structured_text
from lunarops.programs.normal_points_convert import normal_points_convert
from lunarops.programs.registry import ensure_builtin_programs, resolve_program_config
from tests.support.observation import npt_record
from tests.support.samples import crd_text, mini_line

CRD_OPTIONS = {"uncertainty_policy": "binRmsAsSigma"}


def _crd(tmp_path, **changes):
    path = tmp_path / "sample.crd"
    path.write_text(crd_text(**changes), encoding="ascii")
    return path


def _mini(tmp_path, *lines):
    path = tmp_path / "sample.mini"
    path.write_text("\n".join(lines or (mini_line(),)) + "\n", encoding="ascii")
    return path


def test_crd_import_keeps_identifiers_times_and_meteorology(tmp_path):
    record = read_normal_point_source(_crd(tmp_path), crd_options=CRD_OPTIONS).records[0]

    assert (record.station_name, record.station_code, record.reflector_name) == ("APOLLO", "70610", "APOLLO15")
    assert record.round_trip_time_s == pytest.approx(2.500000000123)
    assert record.uncertainty_two_way_s == pytest.approx(12.3456e-12)
    assert record.range_uncertainty_one_way_m == pytest.approx(0.5 * C * 12.3456e-12)
    assert (record.pressure_hpa, record.temperature_k) == pytest.approx((900.123, 280.123))


def test_crd_sessions_crossing_midnight_use_absolute_dates(tmp_path):
    path = _crd(tmp_path, session="2020 1 2 23 58 0 2020 1 3 0 2 0",
                meteorology="20 86340 800 280 50 0\n20 60 900 280 50 0",
                normal_point="11 70 2.500000000123 system 2 300.5 42 12.3456 0 0 0 0 0 4.567")
    record = parse_crd_file(path, uncertainty_policy="binRmsAsSigma", meteorology_max_age_s=30).records[0]

    assert record.transmit_epoch.isot(precision=0) == "2020-01-03T00:01:10"
    assert record.pressure_hpa == 900


def test_crd_rejects_unsupported_epoch_events_and_reports_invalid_records(tmp_path):
    with pytest.raises(ValueError, match="unsupported epoch-event 1"):
        read_normal_point_source(_crd(tmp_path, normal_point="11 100.1 2.5 system 1 300.5 42 12.3 0 0 0 0 0 4.5"),
                                 crd_options=CRD_OPTIONS)
    path = _crd(tmp_path)
    path.write_text(path.read_text().replace("H8\n", "11 invalid invalid system 2 300.5 42 invalid\nH8\n"))
    dataset = read_normal_point_source(path, crd_options=CRD_OPTIONS)

    assert (len(dataset.records), dataset.n_input_records, dataset.n_invalid_records) == (1, 2, 1)
    assert dataset.import_issues[0]["line"] == 8


def test_mini_import_is_only_available_at_the_converter_boundary(tmp_path):
    path = _mini(tmp_path)
    with pytest.raises(ValueError, match=r"\.txt"):
        read_normal_points(path)
    record = read_normal_point_source(path, crd_options=CRD_OPTIONS).records[0]

    assert record.station_name == "APOLLO"
    assert record.round_trip_time_s == pytest.approx(2.5)
    assert record.uncertainty_two_way_s == pytest.approx(12.3e-12)


@pytest.mark.parametrize(("unit", "raw", "kelvin"),
                         [("deciCelsius", 0, 273.15), ("deciCelsius", 200, 293.15), ("deciKelvin", 2932, 293.2)])
def test_mini_temperature_units(tmp_path, unit, raw, kelvin):
    data = parse_mini_file(_mini(tmp_path, mini_line(temperature=raw)), temperature_unit=unit)

    assert data.records[0].temperature_k == pytest.approx(kelvin)
    assert data.import_metadata["temperatureUnit"] == unit


@pytest.mark.parametrize("name", ["canonical.txt", "canonical.txt.gz"])
def test_native_files_round_trip_every_record_field(tmp_path, name):
    original = read_normal_point_source(_crd(tmp_path), crd_options=CRD_OPTIONS)
    target = tmp_path / name

    assert write_normal_points(original, target) == target
    recovered = read_normal_points(target)
    assert (recovered.name, recovered.n_input_records) == (original.name, original.n_input_records)
    assert recovered.records == original.records
    if name.endswith(".gz"):
        with gzip.open(target, "rt") as stream:
            text = stream.read()
    else:
        text = target.read_text()
    assert "APOLLO15" in text and "Apollo%20" not in text


def test_native_writer_only_accepts_txt_files(tmp_path):
    dataset = read_normal_point_source(_crd(tmp_path), crd_options=CRD_OPTIONS)
    for name in ("normalPoints.jsonl", "normalPoints.csv", "normalPoints.llnpt"):
        with pytest.raises(ValueError, match=r"\.txt"):
            write_normal_points(dataset, tmp_path / name)


def test_time_filters_and_merges_keep_the_source_records_unchanged():
    source = NptDataset([npt_record(7), replace(npt_record(9), transmit_epoch=npt_record().transmit_epoch.shifted(60))])
    middle = npt_record().transmit_epoch.shifted(30)
    merged = combine_npt_datasets([source.filter_time(start_time_utc=middle), source.filter_time(end_time_utc=middle)])

    assert [record.index for record in source] == [7, 9]
    assert [record.index for record in merged] == [0, 1]
    merged.records[0].pressure_hpa = 1
    assert source.records[1].pressure_hpa != 1


def test_converter_is_repeatable_inside_its_input_directory(tmp_path):
    ensure_builtin_programs()
    _crd(tmp_path)
    config = resolve_program_config("NormalPointsConvert", {
        "inputFilesNormalPoints": ["."], "datasetName": "campaign", "crdUncertaintyPolicy": "binRmsAsSigma",
        "outputFileNormalPoints": "canonical.txt.gz", "outputFileImportReport": "importReport.txt.gz"})
    context = RunContext(working_dir=tmp_path)

    assert normal_points_convert(config, context) == normal_points_convert(config, context) == tmp_path / "canonical.txt.gz"
    report = read_structured_text(tmp_path / "importReport.txt.gz", "normalPointImportReport")
    assert read_normal_points(tmp_path / "canonical.txt.gz").name == "campaign"
    assert (report["recordCount"], report["invalidRecordCount"]) == (1, 0)


def test_converter_publishes_import_issues_in_its_report(tmp_path):
    ensure_builtin_programs()
    _mini(tmp_path, mini_line(), "invalid record")
    normal_points_convert(resolve_program_config("NormalPointsConvert", {
        "inputFilesNormalPoints": ["sample.mini"], "outputFileNormalPoints": "normalPoints.txt.gz",
        "outputFileImportReport": "importReport.txt.gz"}), RunContext(working_dir=tmp_path))
    report = read_structured_text(tmp_path / "importReport.txt.gz", "normalPointImportReport")

    assert report["invalidRecordCount"] == 1
    assert report["sources"][0]["issues"][0]["line"] == 2
    assert not (tmp_path / "llr_mini_io_warnings.log").exists()
