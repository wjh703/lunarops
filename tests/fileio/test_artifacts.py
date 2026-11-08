"""Native text and binary artifacts: tokens, records, matrices, parameter products, states and reports."""

import os
from copy import deepcopy

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.lsq.products import CovarianceMatrix, ParameterVector
from lunarops.fileio.archive import decode_token, encode_token, sha256_file
from lunarops.fileio.covariance import read_covariance, write_covariance
from lunarops.fileio.numeric_table import (
    read_matrix,
    read_numeric_table,
    read_numeric_table_type,
    write_matrix,
    write_numeric_table,
)
from lunarops.fileio.parameter_vectors import read_parameter_vector, write_parameter_vector
from lunarops.fileio.processing_artifacts import read_processing_state, write_processing_state
from lunarops.fileio.result_table import read_records, write_records
from lunarops.fileio.structured_reports import (
    read_lunar_orbit_metadata,
    read_processing_report,
    write_lunar_orbit_metadata,
    write_processing_report,
)
from lunarops.fileio.yaml_artifact import write_structured_text

NAMES = (ParameterName("A", "position.x"), ParameterName("A", "position.y"))


def test_parameter_names_print_and_parse_with_all_four_fields():
    name = ParameterName(object_name="apollo15", parameter_type="position.x", temporal="trend")
    assert str(name) == "apollo15:position.x:trend:"
    parsed = ParameterName.parse("GRASSE:rangeBias::")
    assert (parsed.object_name, parsed.parameter_type, str(parsed)) == ("GRASSE", "rangeBias", "GRASSE:rangeBias::")


def test_text_tokens_distinguish_a_literal_tilde_from_a_missing_value():
    assert (encode_token(""), decode_token("~"), decode_token(encode_token("~"))) == ("~", "", "~")
    with pytest.raises(ValueError, match="percent escape"):
        decode_token("bad%token")


def test_structured_text_rejects_opaque_objects_and_non_finite_numbers(tmp_path):
    with pytest.raises(TypeError, match="cannot encode"):
        write_structured_text(tmp_path / "opaque.txt", "testArtifact", {"value": object()})
    with pytest.raises(ValueError, match="non-finite"):
        write_structured_text(tmp_path / "nonfinite.txt", "testArtifact", {"value": np.float64(np.inf)})


@pytest.mark.parametrize("suffix", [".txt", ".txt.gz"])
def test_records_keep_header_scalars_types_missing_values_and_signed_zero(tmp_path, suffix):
    fields = (("name", "text"), ("used", "bool"), ("index", "int"), ("range_m", "float"))
    rows = [
        {"name": "a b#~", "used": True, "index": 3, "range_m": -0.0},
        {"name": "~", "used": False, "index": 4, "range_m": None},
    ]
    header = {"frame": "ITRF", "metadata": {"sources": ["a", None], "count": 2}}
    write_records(tmp_path / f"records{suffix}", "testRecords", header, fields, rows)
    recovered_header, recovered = read_records(tmp_path / f"records{suffix}", "testRecords", expected_fields=fields)

    assert (recovered_header, recovered) == (header, rows)
    assert np.signbit(recovered[0]["range_m"])
    assert type(recovered[0]["used"]) is bool and type(recovered[0]["index"]) is int


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("recordCount 1", "recordCount 2"), "declares 2 rows"),
        (("field value float 1", "field value unknown 1"), "Unknown table field type"),
        (("field value float 1", "field value float m"), "has unit"),
        (("header {}", "header []"), "must be a mapping"),
    ],
)
def test_records_reader_rejects_inconsistent_headers(tmp_path, change, message):
    path = tmp_path / "records.txt"
    write_records(path, "testRecords", {}, (("value", "float"),), [{"value": 1.0}])
    path.write_text(path.read_text().replace(*change))
    with pytest.raises(ValueError, match=message):
        read_records(path, "testRecords")


def test_numeric_tables_round_trip_and_check_their_type(tmp_path):
    values = np.arange(12, dtype=float).reshape(4, 3)
    path = write_numeric_table(
        tmp_path / "orbit.dat", "lunarOrbit", ("t", "x", "y"), values, metadata={"frame": "BCRS"}
    )
    artifact_type, columns, restored, metadata = read_numeric_table(path, "lunarOrbit")

    assert (artifact_type, read_numeric_table_type(path), columns, metadata) == (
        "lunarOrbit",
        "lunarOrbit",
        ("t", "x", "y"),
        {"frame": "BCRS"},
    )
    np.testing.assert_array_equal(restored, values)
    with pytest.raises(ValueError, match="Expected"):
        read_numeric_table(path, "lunarAccelerationDiagnostics")


@pytest.mark.parametrize("suffix", [".dat", ".dat.gz"])
@pytest.mark.parametrize(
    ("kind", "values"),
    [("vector", np.array([1.0, 2.0, 3.0])), ("lowerSymmetric", np.array([[2.0, -0.0], [-0.0, 3.0]]))],
)
def test_matrices_round_trip_bit_for_bit(tmp_path, suffix, kind, values):
    path = tmp_path / f"matrix{suffix}"
    write_matrix(path, values, kind=kind)
    np.testing.assert_array_equal(read_matrix(path, expected_kind=kind), values)
    assert np.signbit(read_matrix(path)).tolist() == np.signbit(values).tolist()


def test_a_lower_symmetric_matrix_is_a_single_column_numeric_table(tmp_path):
    values = np.array([[2.0, 0.5], [0.5, 3.0]])
    path = tmp_path / "matrix.dat"
    write_matrix(path, values, kind="lowerSymmetric")
    kind, columns, payload, metadata = read_numeric_table(path)

    assert (kind, columns, metadata) == ("matrix", ("value",), {"kind": "lowerSymmetric", "rows": 2})
    np.testing.assert_array_equal(payload[:, 0], values[np.tril_indices(2)])
    with pytest.raises(ValueError, match="Expected matrix kind"):
        read_matrix(path, expected_kind="vector")


def test_compressed_artifacts_are_byte_reproducible(tmp_path):
    for name in ("first", "second"):
        write_matrix(tmp_path / f"{name}.dat.gz", np.array([[1.0, 0.25], [0.25, 2.0]]), kind="lowerSymmetric")
        write_structured_text(tmp_path / f"{name}.txt.gz", "testArtifact", {"value": 1})
    assert (tmp_path / "first.dat.gz").read_bytes() == (tmp_path / "second.dat.gz").read_bytes()
    assert (tmp_path / "first.txt.gz").read_bytes() == (tmp_path / "second.txt.gz").read_bytes()


def test_parameter_vectors_and_covariances_round_trip_with_names_units_and_three_sigma(tmp_path):
    vector = ParameterVector(NAMES, np.array([1.25, -2.5]), ("m", "m"), np.array([0.3, 0.6]), 3.0)
    covariance = CovarianceMatrix(NAMES, np.array([[1.0, 0.25], [0.25, 4.0]]), ("m", "m"), "posteriorCovariance")
    write_parameter_vector(vector, tmp_path / "solution.txt")
    write_covariance(covariance, tmp_path / "covariance")
    recovered, matrix = read_parameter_vector(tmp_path / "solution.txt"), read_covariance(tmp_path / "covariance")

    assert recovered.parameter_names == NAMES and matrix.parameter_names == NAMES
    np.testing.assert_array_equal(recovered.values, vector.values)
    np.testing.assert_array_equal(recovered.uncertainties, vector.uncertainties)
    assert recovered.uncertainty_sigma_multiplier == 3.0
    text = (tmp_path / "solution.txt").read_text()
    assert '"hasUncertainty": true' in text and '"uncertaintySigmaMultiplier": 3' in text and "hasSigma" not in text
    assert matrix.kind == "posteriorCovariance"
    np.testing.assert_array_equal(matrix.matrix, covariance.matrix)


def test_covariance_reader_rejects_a_foreign_matrix_file(tmp_path):
    write_covariance(CovarianceMatrix(NAMES[:1], np.array([[1.0]]), ("m",)), tmp_path / "covariance")
    write_numeric_table(tmp_path / "covariance" / "covariance.dat.gz", "other", ("value",), [[1.0]])
    with pytest.raises(ValueError, match="Expected 'matrix'"):
        read_covariance(tmp_path / "covariance")


def test_processing_state_round_trips_and_requires_its_fields(tmp_path):
    payload = {
        "fingerprint": "a" * 64,
        "parametrization": {"stationRangeBias": {"values": {"STA": 0.1}}},
        "reflectorPositions": {"REF": [1.0, 2.0, 3.0]},
        "sigmaFactors": {"component": 1.2},
        "weightFactors": {"1": 0.9},
    }
    write_processing_state(tmp_path / "state.txt.gz", payload)
    assert read_processing_state(tmp_path / "state.txt.gz") == payload
    write_processing_state(
        tmp_path / "old.txt",
        {
            **{k: payload[k] for k in ("fingerprint", "parametrization", "reflectorPositions")},
            "scales": {},
            "robustFactors": {},
        },
    )
    with pytest.raises(ValueError, match="missing field"):
        read_processing_state(tmp_path / "old.txt")


REPORT = {
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
            "values": {"station_a": 0.5},
            "intervals": [
                {"station": "station_a", "start": "2020-01-01", "endExclusive": None, "name": "station_a_interval"}
            ],
        },
        "reflectorPosition": {"reflector_a": [1, 2, 3]},
    },
}


@pytest.mark.parametrize("suffix", [".txt", ".txt.gz"])
def test_processing_reports_round_trip_without_mutating_the_payload(tmp_path, suffix):
    payload = deepcopy(REPORT)
    write_processing_report(tmp_path / f"report{suffix}", payload)
    assert read_processing_report(tmp_path / f"report{suffix}") == REPORT
    assert payload == REPORT


def test_orbit_metadata_round_trips(tmp_path):
    payload = {
        "model": "test",
        "timeScale": "TDB",
        "stepSeconds": 60.0,
        "config": {"lunarDegree2Gravity": {"radiusM": 1738000.0}, "bodyForceTerms": ["figure_moon"]},
        "dynamicsBodyNames": ["MOON"],
        "resolvedForceInventory": {"active": ["figure_moon"], "source": "explicit"},
    }
    write_lunar_orbit_metadata(tmp_path / "meta.txt", payload)
    assert read_lunar_orbit_metadata(tmp_path / "meta.txt") == payload


@pytest.mark.parametrize(
    ("writer", "reader"),
    [(write_processing_report, read_lunar_orbit_metadata), (write_lunar_orbit_metadata, read_processing_report)],
)
def test_report_readers_reject_another_artifact_type(tmp_path, writer, reader):
    writer(tmp_path / "report.txt", {"value": 1.0})
    with pytest.raises(ValueError, match="Expected LunarOps"):
        reader(tmp_path / "report.txt")


def test_file_hash_follows_content_changes(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"one")
    first = sha256_file(path)
    assert sha256_file(path) == first
    stat = path.stat()
    path.write_bytes(b"two")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = sha256_file(path)
    path.write_bytes(b"three")  # a size change is detected even with an unchanged modification time
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert len({first, second, sha256_file(path)}) == 3
