"""LlrProcessing end to end on the synthetic observation pipeline."""

from dataclasses import replace

import numpy as np
import pytest

from lunarops.classes.observation import NptDataset
from lunarops.config.context import RunContext
from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
from lunarops.fileio.covariance import read_covariance
from lunarops.fileio.normal_points import write_normal_points
from lunarops.fileio.observation_results import read_observation_results
from lunarops.fileio.parameter_vectors import read_parameter_vector
from lunarops.fileio.processing_artifacts import read_processing_state
from lunarops.fileio.structured_reports import read_processing_report
from lunarops.programs import llr_processing as processing
from lunarops.programs.registry import resolve_program_config
from tests.support.observation import npt_record, observation_processor, static_frames, station_facing_the_moon
from tests.tolerances import COVARIANCE_REL, ESTIMATE_REL, assert_close

CONFIG = {
    "inputFilesNormalPoints": ["normal.txt"],
    "inputFileStationCatalog": "stations.txt",
    "inputFileReflectorCatalog": "reflectors.txt",
    "parametrization": [{"type": "stationRangeBias"}],
    "showProgress": False,
    "varianceComponents": [{"id": "A", "station": "APOLLO", "start": "2020-01-01", "endExclusive": None}],
    "processingSteps": [
        {"type": "screenObservations", "residual": {"maximumAbsoluteM": None}},
        {"type": "estimate", "name": "bias", "estimateVarianceFactors": False, "estimateRobustWeights": False,
         "maxIterationCount": 2},
        {"type": "writeResiduals", "outputFile": "residuals.txt", "outputLevel": "full"},
        {"type": "writeResults", "outputFileSolution": "solution.txt", "outputFileCovariance": "covariance",
         "outputFileReport": "report.txt", "outputFileState": "state.txt"},
    ],
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("processing")
    frames = static_frames()
    processor = observation_processor(frames=frames, station_itrf_m=station_facing_the_moon(frames))
    write_normal_points(NptDataset([npt_record(index) for index in range(6)]), root / "normal.txt")
    write_station_catalog(processor.model_state.station_catalog, root / "stations.txt")
    write_reflector_catalog(processor.model_state.reflector_catalog, root / "reflectors.txt")
    config = resolve_program_config("LlrProcessing", CONFIG)
    # The program's processor factory is its injection point for the synthetic pipeline.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(processing, "build_observation_processor", lambda context, config: processor)
        result = processing.llr_processing(config, RunContext(working_dir=root))
    return root, config, result


def test_resolved_config_keeps_the_processing_plan_at_the_top_level(run):
    _, config, result = run
    assert "processingSteps" in config and "varianceComponents" in config and "processing" not in config
    assert result.publication_quality()["formalProductEligible"]


def test_report_and_state_files_reproduce_the_result(run):
    root, _, result = run
    report = read_processing_report(root / "report.txt")
    state = read_processing_state(root / "state.txt")

    assert (root / "report.txt").read_text().splitlines()[0] == "lunarops processingReport"
    assert report["terminationReason"] == result.termination_reason
    assert report["summary"]["sigma0Post"] == result.sigma0_post
    assert report["observations"][0]["residualSigmaM"] == result.observations[0]["residualSigmaM"]
    assert report["parameters"][0]["finalValue"] == result.parameters[0]["finalValue"]
    assert report["processingSteps"][1]["settings"]["estimate"]["estimateVarianceFactors"] is False
    assert state["parametrization"] == result.state
    for item in result.parameters:
        assert_close(item["referenceValue"] + item["appliedUpdate"], item["finalValue"], ESTIMATE_REL)


def test_residual_file_keeps_native_column_types(run):
    root, _, result = run
    rows = read_observation_results(root / "residuals.txt")

    assert len(rows) == len(result.observations)
    assert all(type(row["light_time_converged"]) is bool and type(row["troposphere_clamped"]) is bool for row in rows)
    assert all(type(row["station_id"]) is str and type(row["sigma0_m"]) is float for row in rows)


def test_solution_and_covariance_files_publish_sigma0_scaled_products(run):
    root, _, result = run
    vector, covariance = read_parameter_vector(root / "solution.txt"), read_covariance(root / "covariance")
    scale2 = 1.0 if result.sigma0_post is None else result.sigma0_post**2

    assert_close(vector.values, list(result.state["stationRangeBias"]["values"].values()), ESTIMATE_REL)
    assert_close(vector.values, result.normals.x0, ESTIMATE_REL)
    assert_close(covariance.matrix, scale2 * result.cofactor, COVARIANCE_REL)
    for sigma0 in (None, 2.0):
        scaled = replace(result, sigma0_post=sigma0)
        sigma = np.sqrt(np.maximum(np.diag(result.cofactor), 0.0)) * (1.0 if sigma0 is None else sigma0)
        assert_close(scaled.covariance_matrix().matrix, (1.0 if sigma0 is None else sigma0**2) * result.cofactor,
                     COVARIANCE_REL)
        # Published uncertainties are three sigma.
        assert_close(scaled.parameter_vector().uncertainties, 3.0 * sigma, COVARIANCE_REL)


def test_formal_products_require_convergence_and_unclamped_troposphere(run):
    _, _, result = run
    with pytest.raises(ValueError, match="diagnostic only"):
        replace(result, converged=False, termination_reason="MAX_ITERATION_COUNT_REACHED").require_formal_product()
    with pytest.raises(ValueError, match="clamped"):
        replace(result, observations=[{**result.observations[0], "troposphereClamped": True}]).require_formal_product()
