"""Public naming, current provenance and documentation regression checks."""

import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from lunarops.classes.dynamics.inertia import LunarDegree2GravityParameters
from lunarops.classes.observation_factory import ensure_registered
from lunarops.config.loader import build_run_plan, load_config_file, run_config_schema
from lunarops.config.lunar_models import LUNAR_CORE_SCHEMA, LUNAR_DEGREE2_SCHEMA
from lunarops.config.registry import class_descriptions, validate_class_config, validate_global_class_configs
from lunarops.config.schema import ConfigSchema, number
from lunarops.fileio.report_schema import LUNAR_ORBIT_METADATA_SCHEMA, PROCESSING_REPORT_SCHEMA, SETTINGS
from lunarops.fileio.structured_reports import (
    read_lunar_orbit_metadata,
    read_processing_report,
    write_lunar_orbit_metadata,
    write_processing_report,
)
from lunarops.fileio.yaml_artifact import write_structured_text
from lunarops.programs.lunar_orbit import _degree2_parameters
from lunarops.programs.registry import ensure_builtin_programs, program_specs, validate_program_config

ROOT = Path(__file__).resolve().parents[1]
ensure_registered()
ensure_builtin_programs()


def test_degree2_defaults_and_explicit_adapter_preserve_all_internal_parameters():
    default = LunarDegree2GravityParameters()
    assert _degree2_parameters(LUNAR_DEGREE2_SCHEMA.resolve({})) == default
    public = {
        "j2Undistorted": 2.1e-4,
        "beta": 6.4e-4,
        "gamma": 2.3e-4,
        "loveK2": 0.03,
        "tidalDelayDays": 0.2,
        "radiusM": 1739000.0,
        "meanMotionRadS": 2.7e-6,
        "attitudeDerivativeStepS": 20.0,
        "includeTidalDeformation": False,
        "includeRotationalDeformation": True,
        "c21Static": 1e-10,
        "s21Static": -2e-10,
        "s22Static": 3e-10,
    }
    expected = LunarDegree2GravityParameters(
        j2_undistorted=2.1e-4,
        beta=6.4e-4,
        gamma=2.3e-4,
        love_k2=0.03,
        tidal_delay_days=0.2,
        radius_m=1739000.0,
        mean_motion_rad_s=2.7e-6,
        attitude_derivative_step_s=20.0,
        include_tidal_deformation=False,
        include_rotational_deformation=True,
        c21_static=1e-10,
        s21_static=-2e-10,
        s22_static=3e-10,
    )
    actual = _degree2_parameters(LUNAR_DEGREE2_SCHEMA.resolve(public))
    assert asdict(actual) == asdict(expected)
    np.testing.assert_array_equal(actual.undistorted_inertia(), expected.undistorted_inertia())


@pytest.mark.parametrize(
    "old,new",
    [
        ("j2_undistorted", "j2Undistorted"),
        ("love_k2", "loveK2"),
        ("tidal_delay_days", "tidalDelayDays"),
        ("radius_m", "radiusM"),
        ("mean_motion_rad_s", "meanMotionRadS"),
        ("attitude_derivative_step_s", "attitudeDerivativeStepS"),
        ("c21_static", "c21Static"),
        ("s21_static", "s21Static"),
        ("s22_static", "s22Static"),
        ("include_tidal_deformation", "includeTidalDeformation"),
        ("include_rotational_deformation", "includeRotationalDeformation"),
    ],
)
@pytest.mark.parametrize("both", [False, True])
def test_degree2_rejects_legacy_keys_and_collisions_with_precise_path(old, new, both):
    config = {old: 0.1}
    if both:
        config[new] = 0.2
    with pytest.raises(ValueError, match=rf"LunarOrbitPropagation.lunarDegree2Gravity.*{old}"):
        LUNAR_DEGREE2_SCHEMA.resolve(config, path="LunarOrbitPropagation.lunarDegree2Gravity")


@pytest.mark.parametrize(
    "field,value",
    [
        ("radiusM", 0),
        ("tidalDelayDays", -1),
        ("loveK2", None),
        ("includeTidalDeformation", 1),
        ("meanMotionRadS", float("nan")),
        ("j2Undistorted", True),
        ("attitudeDerivativeStepS", -1),
        ("gamma", 0.01),
        ("RadiusM", 1),
    ],
)
def test_degree2_rejects_invalid_values(field, value):
    with pytest.raises((ValueError, TypeError), match="config"):
        LUNAR_DEGREE2_SCHEMA.resolve({field: value})


@pytest.mark.parametrize("legacy", ["polar_moment_fraction", "viscosity_per_s", "pressure_spin"])
def test_core_rejects_old_and_unknown_keys(legacy):
    cfg = {
        "polarMomentFraction": 0.0007,
        "flattening": 0.0002,
        "viscosityPerS": 7e-14,
        "pressureSpin": "core",
        legacy: 1,
    }
    with pytest.raises(ValueError, match=rf"lunarRotation.core.*{legacy}"):
        LUNAR_CORE_SCHEMA.resolve(cfg, path="lunarRotation.core")


def test_station_interval_config_keeps_internal_date_attribute_and_rejects_old_key():
    from lunarops.config.registry import create

    cfg = {
        "type": "stationRangeBias",
        "per": "station+interval",
        "intervals": [
            {"station": "APOLLO", "start": "2020-01-01", "endExclusive": "2021-01-01"},
        ],
    }
    block = create("parametrization", cfg)
    assert block.intervals[0].end_exclusive.isoformat() == "2021-01-01"
    cfg["intervals"][0]["end_exclusive"] = "2022-01-01"
    with pytest.raises(ValueError, match=r"intervals\[0\].*end_exclusive"):
        validate_class_config("parametrization", cfg)


def _check_config_schema(description):
    for field in description["fields"]:
        assert re.fullmatch(r"[a-z][A-Za-z0-9]*", field["name"]), field
        for key in ("properties", "itemProperties"):
            if key in field:
                _check_config_schema(field[key])


def test_all_registered_config_schemas_and_catalog_use_canonical_field_names():
    for spec in program_specs():
        _check_config_schema(spec.schema.describe())
    for types in class_descriptions().values():
        for description in types.values():
            _check_config_schema(description)
    _check_config_schema(run_config_schema().describe())
    with pytest.raises(ValueError, match="lowerCamelCase"):
        ConfigSchema((number("bad_name"),))
    from lunarops.programs.registry import get_program

    field = get_program("MassCatalogCreate").spec.schema.field_map["expectedGroupCounts"]
    assert field.describe()["freeformIdentifiers"]
    assert field.validate({"de440_major": 11, "sb441_asteroid": 343}, "groups") == {
        "de440_major": 11,
        "sb441_asteroid": 343,
    }


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yml")), ids=lambda p: p.name)
def test_all_formal_yaml_examples_validate_after_expansion(path):
    plan = build_run_plan(load_config_file(path))
    validate_global_class_configs(plan.shared)
    validate_global_class_configs(plan.observation_model)
    for name, body in plan.calls:
        validate_program_config(name, body)


def _program_examples():
    # Historical audits and API signatures are records, not runnable examples.
    for filename in ("docs/PROGRAMS.md", "docs/INPUTS.md", "README.md"):
        text = (ROOT / filename).read_text()
        pattern = r"(?m)^(`{3}|~{3})yaml\s*\n(.*?)^\1\s*$"
        for match in re.finditer(pattern, text, re.DOTALL):
            value = yaml.safe_load(match[2])
            entries = value.get("programs", []) if isinstance(value, dict) else value if isinstance(value, list) else []
            for entry in entries:
                if isinstance(entry, dict) and "program" in entry:
                    yield filename, text[: match.start()].count("\n") + 1, entry


@pytest.mark.parametrize("filename,line,entry", list(_program_examples()), ids=lambda value: str(value)[:70])
def test_documented_program_examples_validate(filename, line, entry):
    body = {key: value for key, value in entry.items() if key != "program"}
    validate_program_config(entry["program"], body)


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
def test_current_processing_report_roundtrip_preserves_identifiers(tmp_path, suffix):
    payload = canonical_report()
    before = deepcopy(payload)
    path = tmp_path / ("report" + suffix)
    write_processing_report(path, payload)
    assert read_processing_report(path) == payload
    assert payload == before


@pytest.mark.parametrize("version", [1, 3])
@pytest.mark.parametrize("keys", ["canonical", "old"])
def test_processing_report_rejects_noncurrent_versions_regardless_of_names(tmp_path, version, keys):
    path = tmp_path / "report.txt"
    payload = canonical_report() if keys == "canonical" else {"termination_reason": "CONVERGED"}
    write_structured_text(path, "processingReport", payload, version=version)
    with pytest.raises(ValueError, match="expected 2"):
        read_processing_report(path)


def test_report_rejects_mixed_names_bad_types_and_missing_fields(tmp_path):
    path = tmp_path / "report.txt"
    payload = canonical_report()
    with pytest.raises(ValueError, match="termination_reason"):
        write_processing_report(path, {**payload, "termination_reason": "CONVERGED"})
    payload["observations"][0]["residualSigmaM"] = "bad"
    with pytest.raises(TypeError, match="residualSigmaM"):
        write_processing_report(path, payload)
    with pytest.raises(ValueError, match="missing"):
        write_processing_report(path, {})


def test_report_settings_preserve_dynamic_ids_and_values():
    settings = {
        "estimate": {"convergenceThresholdByParametrizationM": {"custom_block": 0.01}},
        "observationScreening": {"residual": {"maximumAbsoluteByStationM": {"station_a": 2.0}}},
    }
    assert SETTINGS.validate(settings) == settings


@pytest.mark.parametrize("version", [1, 3])
def test_orbit_metadata_rejects_noncurrent_versions(tmp_path, version):
    path = tmp_path / "meta.txt"
    write_structured_text(
        path, "lunarOrbitMetadata", {"model": "test", "timeScale": "TDB", "config": {}}, version=version
    )
    with pytest.raises(ValueError, match="expected 2"):
        read_lunar_orbit_metadata(path)


def test_orbit_metadata_current_snapshot_roundtrip(tmp_path):
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


def test_scientific_table_names_and_units_are_preserved(tmp_path):
    from lunarops.fileio.result_table import read_table, write_table

    fields = (("residual_sigma_m", "float"), ("duration_s", "float"), ("wavelength_nm", "float"))
    rows = [{"residual_sigma_m": 0.001, "duration_s": 60.0, "wavelength_nm": 532.0}]
    path = tmp_path / "table.txt"
    write_table(rows, path, artifact_type="observationResult", version=1, fields=fields)
    text = path.read_text()
    for name, unit in (("residual_sigma_m", "m"), ("duration_s", "s"), ("wavelength_nm", "nm")):
        assert f"field {name} float {unit}\n" in text
    assert read_table(path, artifact_type="observationResult", version=1, expected_fields=fields) == rows


def test_report_schema_declares_canonical_fields_types_and_units():
    def check(schema):
        if schema.kind == "object":
            for field in schema.fields:
                assert re.fullmatch(r"[a-z][A-Za-z0-9]*", field.name)
                check(field.schema)
        if schema.item:
            check(schema.item)

    check(PROCESSING_REPORT_SCHEMA)
    check(LUNAR_ORBIT_METADATA_SCHEMA)
    assert PROCESSING_REPORT_SCHEMA.describe()["type"] == "object"


def test_cli_resolves_public_config_and_lists_artifact_units(tmp_path, capsys):
    from lunarops.cli import main

    source = tmp_path / "config.yml"
    source.write_text(
        "variables: {group_id: 3}\nprograms:\n  - program: MassCatalogCreate\n    catalogName: test\n    outputFileMassCatalog: masses.txt\n    expectedGroupCounts: {de440_major: '{group_id}'}\n    bodyMasses: [{bodyId: MOON, ephemerisTarget: 301, gm: 4.9e+12, groups: [de440_major]}]\n"
    )
    assert main(["resolve-config", str(source)]) == 0
    config = yaml.safe_load(capsys.readouterr().out)
    assert config["programs"][0]["expectedGroupCounts"] == {"de440_major": 3}
    assert main(["describe-artifacts"]) == 0
    result = yaml.safe_load(capsys.readouterr().out)
    assert result["processingReport"]["version"] == 2
    fields = {f["name"]: f for f in result["predictionWindow"]["fields"]}
    assert fields["duration_s"]["unit"] == "s"


@pytest.mark.parametrize("degree2", [{"radius_m": 1738000.0}, {"radiusM": 1738000.0, "radius_m": 1.0}])
def test_v2_metadata_config_cannot_hide_legacy_keys(tmp_path, degree2):
    path = tmp_path / "meta.txt"
    write_structured_text(
        path,
        "lunarOrbitMetadata",
        {"model": "test", "timeScale": "TDB", "config": {"lunarDegree2Gravity": degree2}},
        version=2,
    )
    with pytest.raises(ValueError, match=r"lunarOrbitMetadata.config.lunarDegree2Gravity.*radius_m"):
        read_lunar_orbit_metadata(path)


@pytest.mark.parametrize("version", [1, 2])
def test_orbit_metadata_rejects_old_core_snapshot(tmp_path, version):
    path = tmp_path / "meta.txt"
    core = {"polar_moment_fraction": 0.0007, "flattening": 0.0002, "viscosity_per_s": 7e-14, "pressure_spin": "mantle"}
    write_structured_text(
        path,
        "lunarOrbitMetadata",
        {"model": "test", "timeScale": "TDB", "config": {"lunarRotation": {"core": core}}},
        version=version,
    )
    with pytest.raises(ValueError, match="expected 2" if version == 1 else "pressure_spin"):
        read_lunar_orbit_metadata(path)
