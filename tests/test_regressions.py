"""End-to-end behaviour checks kept from earlier review rounds."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_normal_point_io import _write_crd, _write_mini
from test_observation_pipeline import _pipeline, _record

from lunarops.classes.builtin_registry import ensure_registered
from lunarops.classes.frames.earth_orientation import TabulatedEarthOrientation, read_iers_c04
from lunarops.classes.observation.normal_points import NptDataset, combine_npt_datasets
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.loader import build_run_plan, load_config_file
from lunarops.config.registry import validate_global_class_configs
from lunarops.fileio.earth_orientation import (
    load_earth_orientation_parameter,
    read_earth_orientation_parameter,
    write_earth_orientation_parameter,
)
from lunarops.fileio.formats.crd import parse_crd_file
from lunarops.fileio.structured_reports import (
    read_lunar_orbit_metadata,
    read_processing_report,
    write_lunar_orbit_metadata,
    write_processing_report,
)
from lunarops.programs.registry import ensure_builtin_programs, resolve_program_config

ROOT = Path(__file__).resolve().parents[1]
ensure_registered()
ensure_builtin_programs()


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yml")), ids=lambda p: p.name)
def test_all_yaml_configs_validate_after_expansion(path):
    plan = build_run_plan(load_config_file(path))
    validate_global_class_configs(plan.shared)
    validate_global_class_configs(plan.observation_model)
    for name, body in plan.calls:
        resolve_program_config(name, body)


def test_crd_cross_midnight_and_meteorology_use_absolute_dates(tmp_path):
    source = tmp_path / "night.crd"
    _write_crd(source)
    text = source.read_text().replace("2020 1 2 0 0 0 2020 1 2 1 0 0", "2020 1 2 23 58 0 2020 1 3 0 2 0")
    text = text.replace("20 100.123456789 900.123", "20 86340 800")
    text = text.replace("11 100.123456789", "20 60 900 280 50 0\n11 70")
    source.write_text(text, encoding="ascii")
    data = parse_crd_file(source, uncertainty_policy="binRmsAsSigma", meteorology_max_age_s=30)
    assert data.records[0].transmit_epoch.isot(precision=0) == "2020-01-03T00:01:10"
    assert data.records[0].pressure_hpa == 900


@pytest.mark.parametrize(
    "temperature_unit, raw, kelvin",
    [("deciCelsius", 0, 273.15), ("deciCelsius", 200, 293.15), ("deciKelvin", 2932, 293.2)],
)
def test_mini_temperature_units(tmp_path, temperature_unit, raw, kelvin):
    from lunarops.fileio.formats.mini import parse_mini_file

    source = tmp_path / "sample.mini"
    _write_mini(source)
    text = source.read_text()
    source.write_text(text[:62] + f"{raw:4d}" + text[66:])
    data = parse_mini_file(source, temperature_unit=temperature_unit)
    assert data.records[0].temperature_k == pytest.approx(kelvin)
    assert data.import_metadata["temperatureUnit"] == temperature_unit


def test_filter_and_merge_preserve_original_records_and_ids():
    source = NptDataset([_record(7), replace(_record(9), transmit_epoch=_record().transmit_epoch.shifted(60))])
    first = source.filter_time(end_time_utc=_record().transmit_epoch.shifted(30))
    second = source.filter_time(start_time_utc=_record().transmit_epoch.shifted(30))
    merged = combine_npt_datasets([second, first])
    assert [rec.index for rec in source] == [7, 9]
    assert [rec.index for rec in merged] == [0, 1]
    merged.records[0].pressure_hpa = 1
    assert source.records[1].pressure_hpa != 1


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


def test_real_c04_roundtrip_and_prediction_filter(tmp_path):
    source = ROOT.parent / "data/auxiliary/eopc04.1962-now.txt"
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


def canonical_report():
    return {
        "converged": False,
        "publicationQuality": {"formalProductEligible": False, "status": "diagnostic", "reasons": ["MAX_ITERATIONS"]},
        "terminationReason": "MAX_ITERATIONS",
        "settings": {},
        "summary": {},
        "parameters": [{"name": "reflector_x", "unit": "m", "finalValue": 3.0}],
        "observations": [
            {
                "observationId": "obs_1",
                "stationId": "station_a",
                "epochUtc": "2020-01-01T00:00:00",
                "lightTimeConverged": True,
                "aprioriSigmaM": 0.001,
                "currentStateResidualM": 0.002,
                "residualSigmaM": 0.001,
            }
        ],
        "sigmaFactors": {"de440_major": 1.0},
        "state": {
            "stationRangeBias": {
                "per": "station+interval",
                "intervals": [
                    {"station": "station_a", "start": "2020-01-01", "endExclusive": None, "name": "station_a_interval"}
                ],
                "values": {"station_a": 0.5},
            },
            "reflectorPosition": {"reflector_a": [1, 2, 3]},
        },
    }


@pytest.mark.parametrize("suffix", [".txt", ".txt.gz"])
def test_processing_report_roundtrip_preserves_identifiers(tmp_path, suffix):
    payload = canonical_report()
    before = deepcopy(payload)
    path = tmp_path / ("report" + suffix)
    write_processing_report(path, payload)
    assert read_processing_report(path) == payload
    assert payload == before


def test_orbit_metadata_roundtrip(tmp_path):
    payload = {
        "model": "test",
        "timeScale": "TDB",
        "stepSeconds": 60.0,
        "config": {"lunarDegree2Gravity": {"radiusM": 1738000.0}, "bodyForceTerms": ["figure_moon"]},
        "dynamicsBodyNames": ["MOON"],
        "resolvedForceInventory": {"active": ["figure_moon"], "source": "explicit"},
    }
    path = tmp_path / "meta.txt"
    write_lunar_orbit_metadata(path, payload)
    assert read_lunar_orbit_metadata(path) == payload
