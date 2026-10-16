"""Regression cases for the audited scientific input and quality boundaries."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_normal_point_io import _write_crd, _write_mini
from test_observation_pipeline import _pipeline, _record

from lunarops.classes.dynamics.gravity import load_gravity_field
from lunarops.classes.dynamics.integrators import IntegratorSettings
from lunarops.classes.frames.earth_orientation import TabulatedEarthOrientation, read_iers_c04
from lunarops.classes.observation.normal_points import NptDataset, combine_npt_datasets
from lunarops.classes.time import Epoch, TimeScale, TimeScaleConverter
from lunarops.config.context import RunContext, _config_key
from lunarops.config.fingerprints import kernel_manifest, scientific_fingerprint
from lunarops.fileio.earth_orientation import (
    load_earth_orientation_parameter,
    read_earth_orientation_parameter,
    write_earth_orientation_parameter,
)
from lunarops.fileio.formats.crd import parse_crd_file
from lunarops.fileio.formats.normal_point_sources import detect_normal_point_format
from lunarops.fileio.normal_points import read_normal_points, write_normal_points


def crd(path, text=None, **options):
    if text is not None:
        path.write_text(text, encoding="ascii")
    return parse_crd_file(path, uncertainty_policy="binRmsAsSigma", **options)


@pytest.mark.parametrize("event", [0, 1, 3, 4, 5, 6, 7])
def test_crd_rejects_nontransmit_epoch_with_line_and_count(tmp_path, event):
    source = tmp_path / "sample.crd"
    _write_crd(source)
    text = source.read_text().replace("H8\n", f"11 101 2.5 system {event} 300 42 10 0 0 0 0 0 1\nH8\n")
    data = crd(source, text)
    assert (len(data), data.n_input_records, data.n_invalid_records) == (1, 2, 1)
    assert data.import_issues[0]["line"] == 8
    assert str(event) in data.import_issues[0]["reason"]


@pytest.mark.parametrize(
    "change, reason",
    [
        (("H1 CRD 2", "H1 CRD invalid"), "H1"),
        (("H2 APOL 70610 1 1 3 ILRS\n", ""), "H3"),
        (("C0 0 532.123 system\n", ""), "configuration"),
        (("system 2 300.5", "unknown 2 300.5"), "configuration"),
        (("20 100.123456789", "20 invalid"), "meteorolog"),
        (("20 100.123456789", "20 3000"), "meteorolog"),
        (("H8\n", ""), "H8"),
    ],
)
def test_crd_invalid_inputs_do_not_become_valid_numbers(tmp_path, change, reason):
    source = tmp_path / "bad.crd"
    _write_crd(source)
    with pytest.raises(ValueError, match=reason):
        crd(source, source.read_text().replace(*change), meteorology_max_age_s=60)


def test_crd_cross_midnight_and_meteorology_use_absolute_dates(tmp_path):
    source = tmp_path / "night.crd"
    _write_crd(source)
    text = source.read_text().replace("2020 1 2 0 0 0 2020 1 2 1 0 0", "2020 1 2 23 58 0 2020 1 3 0 2 0")
    text = text.replace("20 100.123456789 900.123", "20 86340 800")
    text = text.replace("11 100.123456789", "20 60 900 280 50 0\n11 70")
    data = crd(source, text, meteorology_max_age_s=30)
    assert data.records[0].transmit_epoch.isot(precision=0) == "2020-01-03T00:01:10"
    assert data.records[0].pressure_hpa == 900


def test_crd_requires_explicit_sigma_and_preserves_source_quality(tmp_path):
    source = tmp_path / "sample.crd"
    _write_crd(source)
    with pytest.raises(ValueError, match="explicit uncertainty_policy"):
        parse_crd_file(source)
    data = parse_crd_file(source, uncertainty_policy="fixed", uncertainty_two_way_s=1e-10)
    assert data.records[0].uncertainty_two_way_s == 1e-10
    assert data.import_metadata["sourceQuality"][0]["binRmsPs"] == 12.3456
    target = tmp_path / "canonical.txt"
    write_normal_points(data, target)
    assert read_normal_points(target).import_metadata == data.import_metadata


@pytest.mark.parametrize("bad", ["\xff", "H1 invalid", "", "not a normal point"])
def test_detector_rejects_bad_content_even_with_known_suffix(tmp_path, bad):
    source = tmp_path / "bad.crd"
    source.write_bytes(bad.encode("latin1"))
    with pytest.raises(ValueError, match="line 1"):
        detect_normal_point_format(source)


def test_detector_rejects_wrong_suffix_and_accepts_content(tmp_path):
    source = tmp_path / "sample.crd"
    _write_mini(source)
    with pytest.raises(ValueError, match="suffix expects crd"):
        detect_normal_point_format(source)
    source.rename(tmp_path / "data.unknown")
    assert detect_normal_point_format(tmp_path / "data.unknown") == "mini"


@pytest.mark.parametrize(
    "temperature_unit, raw, kelvin",
    [
        ("deciCelsius", 0, 273.15),
        ("deciCelsius", 200, 293.15),
        ("deciKelvin", 2932, 293.2),
    ],
)
def test_mini_temperature_units_are_explicit(tmp_path, temperature_unit, raw, kelvin):
    from lunarops.fileio.formats.mini import parse_mini_file

    source = tmp_path / "sample.mini"
    _write_mini(source)
    text = source.read_text()
    source.write_text(text[:62] + f"{raw:4d}" + text[66:])
    data = parse_mini_file(source, temperature_unit=temperature_unit)
    assert data.records[0].temperature_k == pytest.approx(kelvin)
    assert data.import_metadata["temperatureUnit"] == temperature_unit


def gravity_text():
    return (
        "begin_of_head\nmodelname test\ngravity_constant 4.9e12\nradius 1738000\n"
        "max_degree 2\nnorm fully_normalized\ntide_system zero_tide\nend_of_head\n"
        "gfc 0 0 1 0\ngfc 2 0 -9e-5 0\ngfc 2 1 0 0\ngfc 2 2 3e-5 0\n"
    )


@pytest.mark.parametrize("record", ["gfct", "trnd", "asin", "acos"])
def test_time_variable_gravity_is_rejected_including_truncated_degrees(tmp_path, record):
    source = tmp_path / "field.gfc"
    source.write_text(gravity_text() + f"{record} 9 0 1 0 2000\n")
    with pytest.raises(ValueError, match="line.*time-variable"):
        load_gravity_field(source, max_degree=2)


@pytest.mark.parametrize(
    "change, reason",
    [
        (("gfc 2 1 0 0\n", ""), "missing coefficients"),
        (("gfc 2 1 0 0\n", "gfc 2 1 0 0\ngfc 2 1 0 0\n"), "duplicate"),
        (("fully_normalized", "unnormalized"), "norm"),
        (("max_degree 2", "max_degree 3"), "max_degree"),
        (("zero_tide", "unknown"), "tide_system"),
        (("gfc 2 1", "gfc 2 3"), "degree/order"),
    ],
)
def test_gravity_contract_rejects_corrupt_static_fields(tmp_path, change, reason):
    source = tmp_path / "field.gfc"
    source.write_text(gravity_text().replace(*change))
    with pytest.raises(ValueError, match=reason):
        load_gravity_field(source)


def test_sparse_gravity_requires_explicit_permission(tmp_path):
    source = tmp_path / "zonal.gfc"
    source.write_text(gravity_text().replace("gfc 2 1 0 0\n", ""))
    data = load_gravity_field(source, allow_missing_coefficients=True)
    assert data.source_metadata["missingCoefficients"] == [(2, 1)]
    assert data.coeffs[0, 2, 1] == 0


def test_fingerprint_tracks_only_declared_inputs_and_calceph_load_order(tmp_path):
    kernels = tmp_path / "kernels"
    kernels.mkdir()
    for name in ("b.bsp", "a.bpc", "a.bsp"):
        (kernels / name).write_bytes(name.encode())
    context = RunContext(
        working_dir=tmp_path,
        shared_class_configs={
            "ephemerides": {
                "type": "calceph",
                "directory": "kernels",
                "lunarRelativisticScaleConvention": "alreadyScaled",
            }
        },
    )

    def fingerprint(**config):
        return scientific_fingerprint(config, context)

    original = fingerprint(outputFilePrediction="out-a", showProgress=True)
    assert original == fingerprint(outputFilePrediction="out-b", showProgress=False)
    (kernels / "a.bsp").write_bytes(b"changed")
    changed = fingerprint()
    assert changed != original
    (kernels / "c.bsp").write_bytes(b"new")
    assert fingerprint() != changed
    assert [item["relativePath"] for item in kernel_manifest(kernels)] == ["a.bsp", "b.bsp", "c.bsp", "a.bpc"]
    (tmp_path / "incidental-label").write_bytes(b"old")
    before = fingerprint(label="incidental-label")
    (tmp_path / "incidental-label").write_bytes(b"new")
    assert before == fingerprint(label="incidental-label")


@pytest.mark.parametrize("config", [{"x": object()}, {"x": np.nan}, {1: "x", "1": "y"}])
def test_unstable_cache_config_is_rejected(config):
    with pytest.raises((ValueError, TypeError)):
        _config_key("test", config)


def test_time_conversion_fails_when_iteration_does_not_converge(monkeypatch):
    converter = TimeScaleConverter()
    values = iter([1.0, 2.0] * 3)
    monkeypatch.setattr(converter, "tdb_minus_tt_s", lambda *args, **kwargs: next(values))
    with pytest.raises(RuntimeError, match="last change"):
        converter.tt2tdb(Epoch.from_calendar(2020, 1, 1, scale=TimeScale.TT))


def test_filter_and_merge_preserve_original_records_and_ids():
    source = NptDataset([_record(7), replace(_record(9), transmit_epoch=_record().transmit_epoch.shifted(60))])
    first = source.filter_time(end_time_utc=_record().transmit_epoch.shifted(30))
    second = source.filter_time(start_time_utc=_record().transmit_epoch.shifted(30))
    merged = combine_npt_datasets([second, first])
    assert [rec.index for rec in source] == [7, 9]
    assert [rec.index for rec in merged] == [0, 1]
    merged.records[0].pressure_hpa = 1
    assert source.records[1].pressure_hpa != 1


@pytest.mark.parametrize("order", [0, 14, 100, True])
def test_unsupported_interpolation_orders_fail(order):
    with pytest.raises(ValueError, match="1, 13"):
        IntegratorSettings(history_interpolation_order=order)


def test_cleanup_closes_remaining_resources_and_preserves_primary_error():
    calls = []

    class Resource:
        def __init__(self, name, bad=False):
            self.name, self.bad = name, bad

        def close(self):
            calls.append(self.name)
            if self.bad:
                raise OSError("cleanup failed")

    context = RunContext()
    context._transient_resources = [Resource("a"), Resource("b", True), Resource("c")]
    with pytest.raises(RuntimeError, match="primary"):
        with context:
            raise RuntimeError("primary")
    assert calls == ["a", "b", "c"]
    assert len(context.cleanup_failures) == 1
    context = RunContext()
    context._transient_resources = [Resource("b", True)]
    with pytest.raises(RuntimeError, match="cleanup"):
        with context:
            pass


def test_dual_leg_event_closure_uses_final_solution():
    from lunarops.classes.observation.light_time import LightTimeRequest

    processor = _pipeline()
    observation = processor.resolver.resolve(_record())
    solution = processor.observation_model.light_time_solver.solve(
        LightTimeRequest.from_station(
            observation.station,
            observation.reflector.moon_fixed_xyz_m,
            observation.transmit_epoch_utc,
            station_key=observation.station_key,
            pressure_hpa=900,
            temperature_k=285,
            relative_humidity_percent=25,
            wavelength_um=0.532,
        )
    )
    assert solution.light_time_converged
    assert max(abs(value) for value in solution.event_closure_residuals_s) < 2e-12


def test_real_c04_native_and_mpi_preserve_quality(tmp_path):
    source = Path(__file__).resolve().parents[2] / "data/auxiliary/eopc04.1962-now.txt"
    if not source.exists():
        pytest.skip("local scientific data unavailable")
    samples = read_iers_c04(source)
    assert samples[0].source == "C04-20" and samples[0].quality == "final"
    target = tmp_path / "eop.txt"
    write_earth_orientation_parameter(samples[:2], target)
    assert read_earth_orientation_parameter(target) == samples[:2]
    table = TabulatedEarthOrientation(samples[:2])
    assert TabulatedEarthOrientation.from_mpi_payload(table.to_mpi_payload()).samples == table.samples
    prediction = replace(samples[1], quality="prediction", source="finals2000A")
    write_earth_orientation_parameter([samples[0], prediction], target)
    table = load_earth_orientation_parameter(target, allow_predicted=False)
    assert len(table.samples) == 1
    with pytest.raises(ValueError, match="outside"):
        table.polar_motion(Epoch(2400000.5, prediction.mjd_utc, TimeScale.UTC))


@pytest.mark.parametrize(
    "bad",
    [
        "prefix 58849 0.1 0.2 0.3",
        "2020 1 1 58850 0.1 0.2 0.3 0 0 0 0 0 0 0 0 0",
        "2020 1 1 58849 0.1 0.2 0.3 0 invalid 0 0 0 0 0 0 0",
    ],
)
def test_c04_rejects_guessing_misalignment_and_missing_values(tmp_path, bad):
    source = tmp_path / "bad-c04.txt"
    source.write_text(bad + "\n")
    with pytest.raises(ValueError, match="line 1"):
        read_iers_c04(source)


def test_clamped_observation_policy_is_explicit(monkeypatch):
    from lunarops.classes.observation.light_time import LightTimeRequest

    processor = _pipeline()
    observation = processor.resolver.resolve(_record())
    solver = processor.observation_model.light_time_solver
    solution = solver.solve(
        LightTimeRequest.from_station(
            observation.station,
            observation.reflector.moon_fixed_xyz_m,
            observation.transmit_epoch_utc,
            station_key=observation.station_key,
            pressure_hpa=900,
            temperature_k=285,
            relative_humidity_percent=25,
            wavelength_um=0.532,
        )
    )
    solution = replace(solution, uplink=replace(solution.uplink, troposphere_elevation_clamped=True))
    monkeypatch.setattr(solver, "solve", lambda request: solution)
    rejected = processor.observation_model.evaluate(
        observation, min_elevation_deg=-90, low_elevation_policy="rejectClamped"
    )
    diagnostic = processor.observation_model.evaluate(
        observation, min_elevation_deg=-90, low_elevation_policy="allowFlagged"
    )
    assert rejected.below_elevation_limit
    assert not diagnostic.below_elevation_limit
    assert diagnostic.equation.troposphere_clamped


def test_predictor_requires_declared_meteorology_strategy():
    from lunarops.cli import _import_programs
    from lunarops.programs.registry import resolve_program_config

    _import_programs()
    config = {
        "inputFileStationCatalog": "s.txt",
        "inputFileReflectorCatalog": "r.txt",
        "outputFilePrediction": "p.txt",
        "outputFileWindows": "w.txt",
        "startTime": "2020-01-01",
        "endTime": "2020-01-01",
        "stationName": "S",
        "reflectorName": "R",
    }
    with pytest.raises(ValueError, match="meteorologyPolicy"):
        resolve_program_config("LlrObservationPrediction", config)
    with pytest.raises(ValueError, match="pressureHpa"):
        resolve_program_config("LlrObservationPrediction", {**config, "meteorologyPolicy": "measured"})
    nominal = resolve_program_config("LlrObservationPrediction", {**config, "meteorologyPolicy": "nominal"})
    assert nominal["pressureHpa"] == 900


def test_explicit_empty_force_list_fails_and_omission_remains_distinct():
    from lunarops.cli import _import_programs
    from lunarops.programs.registry import get_program

    _import_programs()
    config = {
        "ephemerides": {"type": "calceph", "directory": "kernels", "lunarRelativisticScaleConvention": "alreadyScaled"},
        "inputFileMassCatalog": "m.txt",
        "initialTdbJd1": 2451545,
        "durationSeconds": 5400,
        "outputFileOrbit": "o.txt",
        "outputFileAccelerationDiagnostics": "a.txt",
        "outputFileMetadata": "meta.txt",
    }
    schema = get_program("LunarOrbitPropagation").spec.schema
    assert "bodyForceTerms" not in schema.resolve(config)
    with pytest.raises(ValueError, match="bodyForceTerms"):
        schema.resolve({**config, "bodyForceTerms": []})
    assert schema.resolve({**config, "bodyForceTerms": ["point_mass"]})["bodyForceTerms"] == ["point_mass"]
