import numpy as np
import pytest
from typing import Any, cast

import lunarops.cli as cli
from lunarops.classes.time import Epoch, TimeScale
from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.catalogs import ReflectorRecord
from lunarops.config.context import RunContext
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.estimation.observation_equations import FrozenObservationEquations
from lunarops.estimation.parameter_products import CovarianceMatrix, ParameterVector
from lunarops.fileio.adjustment import read_adjustment_state, write_adjustment_state
from lunarops.fileio.archive import decode_token, encode_token
from lunarops.fileio.catalogs import (
    read_reflector_catalog,
    write_reflector_catalog,
)
from lunarops.fileio.matrix import matrix_kind, read_matrix, write_matrix
from lunarops.fileio.normal_equation_file import write_normal_equations
from lunarops.fileio.observation_equation_file import (
    read_observation_equations,
    write_observation_equations,
)
from lunarops.fileio.parameters import (
    read_covariance,
    read_parameter_vector,
    write_covariance,
    write_parameter_vector,
)
from lunarops.fileio.structured_text import read_structured_text, write_structured_text
from lunarops.programs.registry import run_program


@pytest.mark.parametrize("suffix", [".txt", ".txt.gz", ".dat", ".dat.gz"])
@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("dense", np.array([[1.0, 2.0], [3.0, 4.0]])),
        ("vector", np.array([1, 2, 3], dtype=np.int64)),
        ("lowerSymmetric", np.array([[2.0, 0.5], [0.5, 3.0]])),
    ],
)
def test_matrix_encodings_round_trip(tmp_path, suffix, kind, values):
    path = tmp_path / f"matrix{suffix}"
    write_matrix(path, values, kind=kind)

    assert matrix_kind(path) == kind
    assert np.array_equal(read_matrix(path), values)


def test_compressed_artifacts_are_byte_reproducible(tmp_path):
    values = np.array([[1.0, 0.25], [0.25, 2.0]])
    for suffix in (".txt.gz", ".dat.gz"):
        first = tmp_path / f"first{suffix}"
        second = tmp_path / f"second{suffix}"
        write_matrix(first, values, kind="lowerSymmetric")
        write_matrix(second, values, kind="lowerSymmetric")
        assert first.read_bytes() == second.read_bytes()


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
        kind="estimate",
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
    assert recovered_vector.kind == "estimate"
    assert np.allclose(recovered_vector.values, vector.values)
    assert recovered_vector.uncertainties is not None
    assert vector.uncertainties is not None
    assert np.allclose(recovered_vector.uncertainties, vector.uncertainties)
    assert recovered_vector.uncertainty_sigma_multiplier == pytest.approx(3.0)
    vector_text = vector_path.read_text()
    assert "hasUncertainty 1" in vector_text
    assert "uncertaintySigmaMultiplier 3" in vector_text
    assert "hasSigma" not in vector_text
    assert recovered_covariance.parameter_names == names
    assert recovered_covariance.kind == "posteriorCovariance"
    assert np.allclose(recovered_covariance.matrix, covariance.matrix)


def test_covariance_checksum_mismatch_is_rejected(tmp_path):
    name = ParameterName("A", "position.x")
    path = tmp_path / "covariance"
    write_covariance(CovarianceMatrix((name,), np.array([[1.0]]), ("m",)), path)
    payload = path / "covariance.dat.gz"
    payload.write_bytes(payload.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_covariance(path)


def test_normal_equation_addition_aligns_permuted_parameter_names():
    x = ParameterName("test", "x")
    y = ParameterName("test", "y")
    first = NormalEquations.zeros([x, y])
    first.accumulate(np.array([[1.0, 2.0]]), np.array([3.0]), np.array([1.0]))
    second = NormalEquations.zeros([y, x])
    second.accumulate(np.array([[4.0, 5.0]]), np.array([6.0]), np.array([2.0]))

    combined = first.add(second)
    expected_second = np.array([[5.0, 4.0]])
    expected_n = np.array([[1.0, 2.0]]).T @ np.array([[1.0, 2.0]])
    expected_n += 0.25 * expected_second.T @ expected_second
    expected_w = np.array([1.0, 2.0]) * 3.0 + 0.25 * expected_second[0] * 6.0

    assert combined.parameter_names == [x, y]
    assert np.allclose(combined.N, expected_n)
    assert np.allclose(combined.W, expected_w)


def test_normal_equation_addition_rejects_different_model_fingerprints():
    name = ParameterName("test", "x")
    first = NormalEquations.zeros([name], compatibility="a" * 64)
    second = NormalEquations.zeros([name], compatibility="b" * 64)

    with pytest.raises(ValueError, match="incompatible scientific conventions"):
        first.add(second)
    with pytest.raises(ValueError, match="incompatible scientific conventions"):
        first.add(NormalEquations.zeros([name]))


def _frozen_equations() -> FrozenObservationEquations:
    names = (ParameterName("test", "x"), ParameterName("test", "y"))
    epochs = tuple(Epoch.from_isot(f"2020-01-0{day}T00:00:00", scale=TimeScale.UTC) for day in (1, 2, 3))
    return FrozenObservationEquations(
        names,
        ("m", "m"),
        np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]]),
        np.array([2.0, -1.0, 3.0]),
        np.array([1.0, 2.0, 0.5]),
        (10, 11, 12),
        ("a", "a", "b"),
        epochs,
        ("STA", "STA", "STB"),
        ("REF", "REF", "REF"),
        (True, True, False),
        (532.0, 532.0, None),
        {
            "linearization": "fixed",
            "modelFingerprint": "abc",
            "compatibility": "a" * 64,
        },
    )


def test_observation_equation_group_round_trip_and_normal_equivalence(tmp_path):
    frozen = _frozen_equations()
    path = tmp_path / "equations"
    write_observation_equations(frozen, path)
    recovered = read_observation_equations(path)
    direct = frozen.normal_equations()
    persisted = recovered.normal_equations()

    assert recovered.metadata == frozen.metadata
    assert recovered.parameter_names == frozen.parameter_names
    assert np.allclose(recovered.design, frozen.design)
    assert np.allclose(persisted.N, direct.N)
    assert np.allclose(persisted.W, direct.W)
    assert persisted.lPl == pytest.approx(direct.lPl)
    assert persisted.meta["source"] == "FrozenObservationEquations"


def test_adjustment_state_round_trip_is_distinct_from_report(tmp_path):
    path = tmp_path / "state.txt.gz"
    payload = {
        "fingerprint": "a" * 64,
        "parametrization": {"stationRangeBias": {"values": {"STA": 0.1}}},
        "reflectorPositions": {"REF": [1.0, 2.0, 3.0]},
        "scales": {"component": 1.2},
        "robustFactors": {"1": 0.9},
    }

    write_adjustment_state(path, payload)
    assert read_adjustment_state(path) == payload


def test_normals_solve_program_publishes_all_typed_products(tmp_path):
    names = [ParameterName("test", "x"), ParameterName("test", "y")]
    normals = NormalEquations.zeros(names)
    normals.accumulate(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([1.0, 2.0, 3.1]),
        np.ones(3),
    )
    write_normal_equations(normals, tmp_path / "normals")
    cli._import_programs()
    context = RunContext(working_dir=tmp_path)

    solution = cast(
        ParameterVector,
        run_program(
            "NormalsSolve",
            {
                "inputFileNormalEquations": "normals",
                "outputFileSolution": "solution.txt.gz",
                "outputFileCovariance": "covariance",
                "outputFileReport": "solveReport.txt",
            },
            context,
        ),
    )

    assert solution.kind == "correction"
    persisted_solution = read_parameter_vector(tmp_path / "solution.txt.gz")
    persisted_covariance = read_covariance(tmp_path / "covariance")
    assert persisted_solution.parameter_names == tuple(names)
    assert persisted_solution.uncertainty_sigma_multiplier == pytest.approx(3.0)
    assert persisted_solution.uncertainties is not None
    assert np.allclose(
        persisted_solution.uncertainties,
        3.0 * np.sqrt(np.diag(persisted_covariance.matrix)),
    )
    assert persisted_covariance.parameter_names == tuple(names)
    assert (tmp_path / "solveReport.txt").read_text().startswith("lunarops normalEquationSolutionReport")


def test_apply_solution_publishes_catalog_and_model_state(tmp_path):
    catalog = {"REF": ReflectorRecord("Reflector", [10.0, 20.0, 30.0])}
    names = (
        ParameterName("REF", "position.x"),
        ParameterName("REF", "position.y"),
        ParameterName("REF", "position.z"),
        ParameterName("STA", "rangeBias"),
    )
    solution = ParameterVector(
        names,
        np.array([1.0, -2.0, 3.0, 0.25]),
        ("m", "m", "m", "m"),
        kind="correction",
    )
    write_reflector_catalog(catalog, tmp_path / "reflectors.txt")
    write_parameter_vector(solution, tmp_path / "solution.txt")
    cli._import_programs()

    run_program(
        "LlrApplySolution",
        {
            "inputFileSolution": "solution.txt",
            "inputFileReflectorCatalog": "reflectors.txt",
            "outputFileReflectorCatalog": "updatedReflectors.txt",
            "outputFileModelState": "modelState.txt",
        },
        RunContext(working_dir=tmp_path),
    )

    updated = read_reflector_catalog(tmp_path / "updatedReflectors.txt")
    state = cast(dict[str, Any], read_structured_text(tmp_path / "modelState.txt", "llrModelState"))
    assert np.allclose(updated["REF"].moon_fixed_xyz_m, [11.0, 18.0, 33.0])
    assert state["solutionKind"] == "correction"
    assert state["rangeBiasValuesM"]["STA:rangeBias::"] == pytest.approx(0.25)
