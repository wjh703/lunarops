import os

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.parameter_products import CovarianceMatrix, ParameterVector
from lunarops.fileio.archive import decode_token, encode_token
from lunarops.fileio.covariance import read_covariance, write_covariance
from lunarops.fileio.numeric_table import read_matrix, write_matrix
from lunarops.fileio.parameter_vectors import read_parameter_vector, write_parameter_vector
from lunarops.fileio.processing_artifacts import read_processing_state, write_processing_state
from lunarops.fileio.yaml_artifact import write_structured_text


@pytest.mark.parametrize("suffix", [".dat", ".dat.gz"])
@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("vector", np.array([1.0, 2.0, 3.0])),
        ("lowerSymmetric", np.array([[2.0, 0.5], [0.5, 3.0]])),
    ],
)
def test_matrix_encodings_round_trip(tmp_path, suffix, kind, values):
    path = tmp_path / f"matrix{suffix}"
    write_matrix(path, values, kind=kind)

    assert np.array_equal(read_matrix(path, expected_kind=kind), values)


def test_compressed_artifacts_are_byte_reproducible(tmp_path):
    values = np.array([[1.0, 0.25], [0.25, 2.0]])
    first = tmp_path / "first.dat.gz"
    second = tmp_path / "second.dat.gz"
    write_matrix(first, values, kind="lowerSymmetric")
    write_matrix(second, values, kind="lowerSymmetric")
    assert first.read_bytes() == second.read_bytes()
    first_text = tmp_path / "first.txt.gz"
    second_text = tmp_path / "second.txt.gz"
    write_structured_text(first_text, "testArtifact", {"value": 1})
    write_structured_text(second_text, "testArtifact", {"value": 1})
    assert first_text.read_bytes() == second_text.read_bytes()


def test_text_tokens_distinguish_literal_tilde_from_missing_value():
    assert encode_token("") == "~"
    assert decode_token("~") == ""
    assert decode_token(encode_token("~")) == "~"
    with pytest.raises(ValueError, match="percent escape"):
        decode_token("bad%token")


def test_structured_text_rejects_opaque_python_objects(tmp_path):
    with pytest.raises(TypeError, match="cannot encode"):
        write_structured_text(tmp_path / "opaque.txt", "testArtifact", {"value": object()})
    with pytest.raises(ValueError, match="non-finite"):
        write_structured_text(tmp_path / "nonfinite.txt", "testArtifact", {"value": np.float64(np.inf)})


def test_parameter_vector_and_covariance_round_trip_with_names_and_units(tmp_path):
    names = (ParameterName("A", "position.x"), ParameterName("A", "position.y"))
    vector = ParameterVector(
        parameter_names=names,
        values=np.array([1.25, -2.5]),
        units=("m", "m"),
        uncertainties=np.array([0.3, 0.6]),
        uncertainty_sigma_multiplier=3.0,
    )
    covariance = CovarianceMatrix(names, np.array([[1.0, 0.25], [0.25, 4.0]]), ("m", "m"), "posteriorCovariance")
    vector_path = tmp_path / "solution.txt"
    covariance_path = tmp_path / "covariance"

    write_parameter_vector(vector, vector_path)
    write_covariance(covariance, covariance_path)
    recovered_vector = read_parameter_vector(vector_path)
    recovered_covariance = read_covariance(covariance_path)

    assert recovered_vector.parameter_names == names
    assert np.allclose(recovered_vector.values, vector.values)
    assert recovered_vector.uncertainties is not None
    assert vector.uncertainties is not None
    assert np.allclose(recovered_vector.uncertainties, vector.uncertainties)
    assert recovered_vector.uncertainty_sigma_multiplier == pytest.approx(3.0)
    vector_text = vector_path.read_text()
    assert '"hasUncertainty": true' in vector_text
    assert '"uncertaintySigmaMultiplier": 3' in vector_text
    assert "hasSigma" not in vector_text
    assert recovered_covariance.parameter_names == names
    assert recovered_covariance.kind == "posteriorCovariance"
    assert np.allclose(recovered_covariance.matrix, covariance.matrix)


def test_covariance_reader_rejects_wrong_matrix_type(tmp_path):
    from lunarops.fileio.numeric_table import write_numeric_table

    name = ParameterName("A", "position.x")
    path = tmp_path / "covariance"
    write_covariance(CovarianceMatrix((name,), np.array([[1.0]]), ("m",)), path)
    write_numeric_table(path / "covariance.dat.gz", "other", ("value",), [[1.0]])
    with pytest.raises(ValueError, match="Expected 'matrix'"):
        read_covariance(path)


def test_processing_state_round_trip_is_distinct_from_report(tmp_path):
    path = tmp_path / "state.txt.gz"
    payload = {
        "fingerprint": "a" * 64,
        "parametrization": {"stationRangeBias": {"values": {"STA": 0.1}}},
        "reflectorPositions": {"REF": [1.0, 2.0, 3.0]},
        "sigmaFactors": {"component": 1.2},
        "weightFactors": {"1": 0.9},
    }

    write_processing_state(path, payload)
    assert read_processing_state(path) == payload


def test_processing_state_is_validated_when_read(tmp_path):
    payload = {
        "fingerprint": "a" * 64,
        "parametrization": {},
        "reflectorPositions": {},
        "scales": {"component": 1.2},
        "robustFactors": {"1": 0.9},
    }

    path = tmp_path / "state.txt"
    write_processing_state(path, payload)
    with pytest.raises(ValueError, match="missing field"):
        read_processing_state(path)


def test_file_hash_cache_tracks_size_and_mtime(tmp_path):
    from lunarops.fileio.archive import _sha256_file, sha256_file

    path = tmp_path / "input"
    path.write_bytes(b"one")
    first = sha256_file(path)
    info = _sha256_file.cache_info()
    assert sha256_file(path) == first
    assert _sha256_file.cache_info().hits == info.hits + 1
    stat = path.stat()
    path.write_bytes(b"two")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = sha256_file(path)
    assert first != second
    assert _sha256_file.cache_info().misses == info.misses + 1
    stat = path.stat()
    path.write_bytes(b"three")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert sha256_file(path) != second
    assert _sha256_file.cache_info().misses == info.misses + 2


@pytest.mark.parametrize("suffix", [".txt", ".txt.gz"])
def test_records_preserve_header_scalar_types_and_missing_values(tmp_path, suffix):
    from lunarops.fileio.result_table import read_records, write_records

    path = tmp_path / ("records" + suffix)
    fields = (("name", "text"), ("used", "bool"), ("index", "int"), ("range_m", "float"))
    rows = [{"name": "a b#~", "used": True, "index": 3, "range_m": -0.0},
            {"name": "~", "used": False, "index": 4, "range_m": None}]
    header = {"frame": "ITRF", "metadata": {"sources": ["a", None], "count": 2}}
    write_records(path, "testRecords", header, fields, rows)
    recovered_header, recovered = read_records(path, "testRecords", expected_fields=fields)
    assert recovered_header == header
    assert recovered == rows
    assert np.signbit(recovered[0]["range_m"])
    assert type(recovered[0]["used"]) is bool
    assert type(recovered[0]["index"]) is int


@pytest.mark.parametrize("change, message", [
    (("recordCount 1", "recordCount 2"), "declares 2 rows"),
    (("field value float 1", "field value unknown 1"), "Unknown table field type"),
    (("field value float 1", "field value float m"), "has unit"),
    (("header {}", "header []"), "must be a mapping"),
])
def test_records_reader_rejects_invalid_headers_and_schema(tmp_path, change, message):
    from lunarops.fileio.result_table import read_records, write_records

    path = tmp_path / "records.txt"
    write_records(path, "testRecords", {}, (("value", "float"),), [{"value": 1.0}])
    path.write_text(path.read_text().replace(*change))
    with pytest.raises(ValueError, match=message):
        read_records(path, "testRecords")


def test_lower_triangle_is_a_single_column_numeric_table(tmp_path):
    from lunarops.fileio.numeric_table import read_numeric_table

    path = tmp_path / "matrix.dat"
    values = np.array([[2.0, -0.0], [-0.0, 3.0]])
    write_matrix(path, values, kind="lowerSymmetric")
    kind, columns, payload, metadata = read_numeric_table(path)
    assert kind == "matrix" and columns == ("value",)
    assert metadata == {"kind": "lowerSymmetric", "rows": 2}
    assert payload[:, 0].tobytes() == values[np.tril_indices(2)].tobytes()
    assert read_matrix(path).tobytes() == values.tobytes()
    with pytest.raises(ValueError, match="Expected matrix kind"):
        read_matrix(path, expected_kind="vector")
