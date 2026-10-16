"""Current-contract acceptance and rejection cases from T01–T18."""

import inspect
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_normal_point_io import _write_crd, _write_mini
from test_second_review_contracts import canonical_report

from lunarops.fileio.structured_reports import (
    read_lunar_orbit_metadata,
    read_processing_report,
    write_lunar_orbit_metadata,
    write_processing_report,
)
from lunarops.fileio.yaml_artifact import write_structured_text


@pytest.mark.parametrize("boundary", ["reader", "writer"])
@pytest.mark.parametrize(
    "change,path",
    [
        ("emptyParameter", "parameters"),
        ("emptyObservation", "observations"),
        ("nullIdentity", "observationId"),
        ("zeroSigma", "aprioriSigmaM"),
        ("negativeSigma", "residualSigmaM"),
        ("missingResidual", "currentStateResidualM"),
        ("oldField", "termination_reason"),
        ("mixedField", "termination_reason"),
        ("nullConverged", "converged"),
        ("falseQuality", "publicationQuality"),
        ("badEpoch", "epoch"),
        ("duplicateObservation", "duplicate observationId"),
        ("duplicateParameter", "duplicate names"),
        ("wrongNames", "parameterNames"),
    ],
)
def test_report_rejects_invalid_current_content(tmp_path, boundary, change, path):
    report = canonical_report()
    if change == "emptyParameter":
        report["parameters"] = [{}]
    elif change == "emptyObservation":
        report["observations"] = [{}]
    elif change == "nullIdentity":
        report["observations"][0]["observationId"] = None
    elif change == "zeroSigma":
        report["observations"][0]["aprioriSigmaM"] = 0.0
    elif change == "negativeSigma":
        report["observations"][0]["residualSigmaM"] = -0.01
    elif change == "missingResidual":
        del report["observations"][0]["currentStateResidualM"]
    elif change == "oldField":
        report["termination_reason"] = report.pop("terminationReason")
    elif change == "mixedField":
        report["termination_reason"] = "CONVERGED"
    elif change == "nullConverged":
        report["converged"] = None
    elif change == "falseQuality":
        report["publicationQuality"]["formalProductEligible"] = True
    elif change == "badEpoch":
        report["observations"][0]["epochUtc"] = "not-an-epoch"
    elif change == "duplicateObservation":
        report["observations"] *= 2
    elif change == "duplicateParameter":
        report["parameters"] *= 2
    elif change == "wrongNames":
        report["parameterNames"] = ["another_parameter"]
    target = tmp_path / "report.txt"
    if boundary == "reader":
        write_structured_text(target, "processingReport", report, version=2)
    with pytest.raises((ValueError, TypeError), match=path if change != "badEpoch" else None):
        read_processing_report(target) if boundary == "reader" else write_processing_report(target, report)


def test_failed_light_time_is_diagnostic_and_cannot_be_published(tmp_path):
    report = canonical_report()
    report["observations"][0].update(lightTimeConverged=False, currentStateResidualM=None, residualSigmaM=None)
    target = tmp_path / "diagnostic.txt"
    write_processing_report(target, report)
    assert read_processing_report(target) == report
    report.update(converged=True, terminationReason="CONVERGED")
    report["publicationQuality"] = {"formalProductEligible": True, "status": "passed", "reasons": []}
    report["parameters"][0]["formalUncertaintyM"] = 0.001
    with pytest.raises(ValueError, match="lightTimeConverged"):
        write_processing_report(target, report)


def test_valid_formal_product_and_diagnostic_nullability(tmp_path):
    report = canonical_report()
    report.update(converged=True, terminationReason="CONVERGED")
    report["publicationQuality"] = {"formalProductEligible": True, "status": "passed", "reasons": []}
    report["parameters"][0]["formalUncertaintyM"] = 0.001
    other = deepcopy(report["observations"][0])
    other["observationId"] = "obs_2"
    report["observations"].append(other)
    report["summary"] = {"rank": 1, "observationCount": 2, "sigma0Post": 1.0}
    target = tmp_path / "formal.txt"
    write_processing_report(target, report)
    assert read_processing_report(target) == report
    report["parameters"][0]["formalUncertaintyM"] = None
    with pytest.raises(ValueError, match="formalUncertaintyM"):
        write_processing_report(target, report)


@pytest.mark.parametrize(
    "config,path",
    [
        ({"lunarDegree2Gravity": {"radius_m": 1738000.0}}, "radius_m"),
        ({"lunarRotation": {"core": {"pressure_spin": "core"}}}, "pressure_spin"),
        ({"solarRelativistic": {"include_lense_thirring": True}}, "include_lense_thirring"),
        ({"earthTide": {"tau_orb_days": [0, 0, 0]}}, "tau_orb_days"),
        ({"ephemerides": "calceph"}, "Class"),
        ({"ephemerides": {"type": "CALCEPH", "directory": "kernels"}}, "CALCEPH"),
    ],
)
@pytest.mark.parametrize("boundary", ["reader", "writer"])
def test_metadata_cannot_hide_old_configuration(tmp_path, config, path, boundary):
    target = tmp_path / "meta.txt"
    payload = {"model": "test", "timeScale": "TDB", "config": config}
    if boundary == "reader":
        write_structured_text(target, "lunarOrbitMetadata", payload, version=2)
    with pytest.raises((ValueError, TypeError, KeyError), match=path if path != "Class" else None):
        read_lunar_orbit_metadata(target) if boundary == "reader" else write_lunar_orbit_metadata(target, payload)


def test_metadata_class_snapshot_keeps_only_supplied_fields(tmp_path):
    payload = {
        "model": "test",
        "timeScale": "TDB",
        "config": {"ephemerides": {"type": "calceph", "directory": "kernels"}},
    }
    target = tmp_path / "meta.txt"
    write_lunar_orbit_metadata(target, payload)
    assert read_lunar_orbit_metadata(target) == payload


@pytest.mark.parametrize("version", [1, 2, 3])
def test_eop_requires_current_version_and_provenance_header(tmp_path, version):
    from lunarops.fileio.earth_orientation import FIELDS, read_earth_orientation_parameter

    target = tmp_path / "eop.txt"
    target.write_text(
        f"lunarops earthOrientationParameter version={version}\nrecordCount 1\nfields "
        + " ".join(FIELDS)
        + "\ndata\n60000 0.1 0.2 -0.3 0 0 0\n"
    )
    with pytest.raises(ValueError, match="header" if version == 2 else "expected 2"):
        read_earth_orientation_parameter(target)


def test_eop_v2_requires_provenance_on_every_row_and_roundtrips(tmp_path):
    from lunarops.classes.frames import EarthOrientationSample
    from lunarops.fileio.earth_orientation import read_earth_orientation_parameter, write_earth_orientation_parameter

    sample = EarthOrientationSample(60000.0, 0.1, 0.2, -0.3, source="C04", quality="final", source_sha256="a" * 64)
    target = tmp_path / "eop.txt"
    write_earth_orientation_parameter([sample], target)
    assert read_earth_orientation_parameter(target) == (sample,)
    lines = target.read_text().splitlines()
    lines[-1] = " ".join(lines[-1].split()[:7])
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="expected 10"):
        read_earth_orientation_parameter(target)


@pytest.mark.parametrize("change,message", [("v1", "line 1"), ("station", "line 2"), ("target", "line 3")])
def test_crd_rejects_version_and_identity_contradictions_immediately(tmp_path, change, message):
    from lunarops.fileio.formats.crd import parse_crd_sessions

    target = tmp_path / "points.crd"
    _write_crd(target)
    old, new = {
        "v1": ("H1 CRD 2", "H1 CRD 1"),
        "station": ("APOL 70610", "APOL 01910"),
        "target": ("H3 APOLLO15", "H3 A15"),
    }[change]
    target.write_text(target.read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        parse_crd_sessions(target)


def test_crd_missing_snr_is_rejected_as_v2_row(tmp_path):
    from lunarops.fileio.formats.crd import parse_crd_sessions

    target = tmp_path / "points.crd"
    _write_crd(target)
    target.write_text(target.read_text().replace("0 0 0 0 0 4.567", "0 0 0 0 0"))
    (session,) = parse_crd_sessions(target)
    assert session.input_record_count == 1
    assert not session.normal_points
    assert "exactly 14 fields including SNR" in session.import_issues[0]["reason"]


def test_mini_crd2_and_native_v1_are_separate_current_channels(tmp_path):
    from lunarops.fileio.formats.normal_point_sources import read_normal_point_source
    from lunarops.fileio.normal_points import read_normal_points, write_normal_points

    for kind, write in [("crd", _write_crd), ("mini", _write_mini)]:
        source = tmp_path / f"points.{kind}"
        write(source)
        data = read_normal_point_source(source, crd_options={"uncertainty_policy": "binRmsAsSigma"})
        target = tmp_path / f"{kind}.txt"
        write_normal_points(data, target)
        assert target.read_text().splitlines()[0] == "lunarops normalPoint version=1"
        restored = read_normal_points(target)
        assert restored.records == data.records


@pytest.mark.parametrize("name", ["lunarorbit", "LUNARORBIT", " LunarOrbit"])
def test_program_names_are_exact(name):
    from lunarops.programs.registry import get_program

    with pytest.raises(KeyError):
        get_program(name)


@pytest.mark.parametrize(
    "cfg", ["mendesPavlis", {"type": "MENDESPAVLIS"}, {"type": "mendespavlis"}, {"type": " mendesPavlis"}, {}]
)
def test_class_declarations_have_one_exact_object_form(cfg):
    from lunarops.classes.observation_factory import ensure_registered
    from lunarops.config.registry import validate_class_config

    ensure_registered()
    with pytest.raises((TypeError, ValueError, KeyError)):
        validate_class_config("troposphere", cfg)


def test_registration_has_one_scope_and_program_declaration_api():
    from lunarops.config.registry import register_factory
    from lunarops.programs.registry import program

    assert "global_scope" not in inspect.signature(register_factory).parameters
    assert tuple(inspect.signature(program).parameters) == ("spec",)
    with pytest.raises(TypeError):
        register_factory("x", "x", lambda *_: None, global_scope=True)
    with pytest.raises(TypeError):
        program(name="Shortcut")


def test_explicit_mpi_rank1_constructor_and_cli_fail(monkeypatch):
    from test_mpi_spec_broadcast import _FakeComm, _FakeMPI

    import lunarops.cli as cli
    import lunarops.parallel.mpi as mpi

    monkeypatch.setattr(mpi, "mpi_comm_world", lambda: _FakeMPI())
    with pytest.raises(RuntimeError, match="at least two ranks"):
        mpi.MpiRuntime(_FakeComm(size=1))
    monkeypatch.setattr(mpi, "MpiRuntime", lambda: SimpleNamespace(size=1, is_master=True))
    with pytest.raises(RuntimeError, match="at least two ranks"):
        cli.cmd_run(SimpleNamespace(mpi=True, config="unused.yml", set=[], working_dir="."))


def test_generic_artifact_helpers_require_explicit_version():
    from lunarops.fileio.archive import atomic_text_writer, parse_header
    from lunarops.fileio.yaml_artifact import read_structured_text

    for function in (atomic_text_writer, write_structured_text, read_structured_text):
        assert inspect.signature(function).parameters["version"].default is inspect.Parameter.empty
    assert "supported_versions" not in inspect.signature(parse_header).parameters


def test_native_identity_source_codes_cannot_contradict_names(tmp_path):
    from dataclasses import replace

    from lunarops.fileio.formats.normal_point_sources import read_normal_point_source

    source = tmp_path / "points.crd"
    _write_crd(source)
    record = read_normal_point_source(source, crd_options={"uncertainty_policy": "binRmsAsSigma"}).records[0]
    with pytest.raises(ValueError, match="station_name contradicts"):
        replace(record, station_code="01910")
    with pytest.raises(ValueError, match="reflector_name contradicts"):
        replace(record, reflector_code="0")
    with pytest.raises(ValueError, match="exact strings"):
        replace(record, station_code=" 70610")


@pytest.mark.parametrize("name", ["moon", "SSB", "EMB", " EARTH", "NAIF:0301", "NAIF:+301", "NAIF: 301"])
def test_body_identity_uses_exact_names_or_naif_ids(name):
    from lunarops.classes.ephemerides.body_ids import canonical_body_id

    with pytest.raises(ValueError):
        canonical_body_id(name)


def test_summary_cannot_inflate_formal_redundancy(tmp_path):
    report = canonical_report()
    report.update(converged=True, terminationReason="CONVERGED")
    report["publicationQuality"] = {"formalProductEligible": True, "status": "passed", "reasons": []}
    report["parameters"][0]["formalUncertaintyM"] = 0.001
    report["summary"] = {"rank": 1, "observationCount": 100, "sigma0Post": 1.0}
    with pytest.raises(ValueError, match="observationCount"):
        write_processing_report(tmp_path / "report.txt", report)


def test_fixed_manifest_fields_cannot_use_opaque_payloads(tmp_path):
    report = canonical_report()
    report["scientificInputs"] = {
        "resources": {
            "models.ephemerides": [
                {"path": "kernels", "kernels": [{"relativePath": "de440.bsp", "sha256": "a" * 64, "loadOrder": 0}]}
            ]
        },
        "versions": {"calcephpy": "5.0.0"},
        "codeSha256": "b" * 64,
    }
    target = tmp_path / "report.txt"
    write_processing_report(target, report)
    assert read_processing_report(target) == report
    report["scientificInputs"]["resources"]["models.ephemerides"][0]["kernels"][0]["load_order"] = 0
    with pytest.raises(ValueError, match="load_order"):
        write_processing_report(target, report)


def test_ast_guard_distinguishes_science_arrays_and_historical_hooks():
    from scripts.check_current_contract import check_source, check_tests

    assert check_source("def prepare(source_names): return source_names.index('SUN')", "lunarops/physics.py") == []
    assert check_source("read_structured_text(path, 'processingReport')", "lunarops/reader.py")
    assert check_source("schema.convert(value, source_names=True)", "lunarops/reader.py")
    assert check_source("def reader(supported_versions): pass", "lunarops/reader.py")
    source = "def test_report():\n write_structured_text(path, 'processingReport', {}, version=1)\n read_processing_report(path)\n"
    assert check_tests(source, "tests/test_report.py")
    source = source.replace(
        " read_processing_report(path)", " with pytest.raises(ValueError):\n  read_processing_report(path)"
    )
    assert check_tests(source, "tests/test_report.py") == []
