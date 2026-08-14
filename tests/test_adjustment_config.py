from copy import deepcopy
from pathlib import Path

import pytest

from lunarops.config.loader import load_config_file
from lunarops.estimation.adjustment_config import parse_adjustment_plan


def _component(component_id="A"):
    return {"id": component_id, "station": "STA_A", "start": "2020-01-01", "endExclusive": None}


def _screen():
    return {"type": "screenObservations"}


def _config():
    return {
        "varianceComponents": [_component()],
        "processingSteps": [
            _screen(),
            {"type": "estimate", "name": "joint"},
            {"type": "writeResults", "outputFileReport": "report.txt"},
        ],
    }


def test_canonical_schema_maps_to_typed_plan():
    config = _config()
    config.update(
        {
            "processingSteps": [
                {
                    "type": "screenObservations",
                    "residual": {"maximumAbsoluteM": 3.0},
                    "reportedSigma": {
                        "minimumOneWayM": 0.002,
                        "minimumFractionOfGroupMedian": 0.1,
                    },
                },
                {"type": "selectParametrizations", "parametrizations": ["offset"]},
                {
                    "type": "estimate",
                    "name": "offset",
                    "maxIterationCount": 4,
                    "convergenceThreshold": 0.001,
                    "convergenceThresholdByParametrizations": {"offset": 0.004},
                    "computeResiduals": True,
                    "estimateVarianceFactors": False,
                    "estimateRobustWeights": True,
                    "robustWeighting": {"model": "igg3", "k0": 1.2, "k1": 5.0},
                },
                {"type": "writeResults", "outputFileReport": "report.txt"},
            ],
        }
    )
    plan = parse_adjustment_plan(config)
    assert plan.settings.accuracy_screening.minimum_one_way_m == pytest.approx(0.002)
    estimate = plan.processing_steps[2]
    assert estimate.robust_weighting.k1 == pytest.approx(5.0)
    settings = estimate.apply(plan.settings)
    assert settings.adjustment.max_iteration_count == 4
    assert settings.adjustment.convergence_threshold_by_parametrization_m == {"offset": 0.004}
    assert not settings.adjustment.adjust_sigma0
    assert settings.adjustment.compute_weights


def test_direct_rejection_uses_k0_only():
    config = _config()
    estimate = config["processingSteps"][1]
    estimate["robustWeighting"] = {"model": "directRejection", "k0": 3.0}
    plan = parse_adjustment_plan(config)
    assert plan.processing_steps[1].robust_weighting.k1 is None
    estimate["robustWeighting"]["k1"] = 6.0
    with pytest.raises(ValueError, match="uses k0 only"):
        parse_adjustment_plan(config)


@pytest.mark.parametrize("section", ["adjustment", "accuracyScreening", "robustWeights"])
def test_removed_top_level_sections_are_rejected(section):
    config = _config()
    config[section] = {}
    with pytest.raises(ValueError, match="Obsolete adjustment section"):
        parse_adjustment_plan(config)


@pytest.mark.parametrize("section", ["vce", "robustEstimation"])
def test_old_section_names_are_not_accepted(section):
    config = _config()
    config[section] = {"components": [_component()]} if section == "vce" else {"model": "igg3"}
    with pytest.raises(ValueError, match="Obsolete adjustment section"):
        parse_adjustment_plan(config)


def test_component_schema_and_duplicate_estimate_names_are_strict():
    config = _config()
    config["varianceComponents"][0]["station_system"] = "A"
    with pytest.raises(ValueError, match="unknown key"):
        parse_adjustment_plan(config)
    config = _config()
    config["processingSteps"] = [
        _screen(),
        {"type": "estimate", "name": "joint"},
        {"type": "estimate", "name": "joint"},
        {"type": "writeResults", "outputFileReport": "report.txt"},
    ]
    with pytest.raises(ValueError, match="names must be unique"):
        parse_adjustment_plan(config)


def test_obsolete_bias_initialization_is_rejected():
    config = _config()
    config["initialization"] = {"biasWeightCap": 1.0e12, "biasMaximumIterations": 30}
    with pytest.raises(ValueError, match="Obsolete adjustment section.*initialization"):
        parse_adjustment_plan(config)


def test_estimate_weight_updates_require_residuals():
    config = _config()
    config["processingSteps"] = [
        _screen(),
        {
            "type": "estimate",
            "name": "joint",
            "computeResiduals": False,
            "estimateRobustWeights": True,
        },
        {"type": "writeResults", "outputFileReport": "report.txt"},
    ]
    with pytest.raises(ValueError, match="require computeResiduals=true"):
        parse_adjustment_plan(config)


@pytest.mark.parametrize("obsolete_key", ["enable", "disable"])
def test_select_parametrizations_rejects_enable_disable(obsolete_key):
    config = _config()
    config["processingSteps"] = [
        _screen(),
        {"type": "selectParametrizations", obsolete_key: ["offset"]},
        {"type": "estimate", "name": "joint"},
        {"type": "writeResults", "outputFileReport": "report.txt"},
    ]
    with pytest.raises(ValueError, match="unknown key"):
        parse_adjustment_plan(config)


def test_screen_observations_must_be_unique_and_first():
    config = _config()
    config["processingSteps"] = [
        {"type": "estimate", "name": "joint"},
        _screen(),
        {"type": "writeResults", "outputFileReport": "report.txt"},
    ]
    with pytest.raises(ValueError, match="must start with exactly one"):
        parse_adjustment_plan(config)

    config["processingSteps"] = [
        _screen(),
        _screen(),
        {"type": "estimate", "name": "joint"},
        {"type": "writeResults", "outputFileReport": "report.txt"},
    ]
    with pytest.raises(ValueError, match="must start with exactly one"):
        parse_adjustment_plan(config)


def test_detailed_config_uses_canonical_schema():
    root = Path(__file__).resolve().parents[1]
    config = load_config_file(root / "configs" / "lunarops_reflector_bias_adjustment_detailed.yml")
    program = next(item for item in config["programs"] if item.get("program") == "LlrProcessing")
    plan = parse_adjustment_plan(deepcopy(program))
    estimates = [step for step in plan.processing_steps if getattr(step, "name", None)]
    assert [step.name for step in estimates] == ["reflector", "bias", "joint"]
    assert len(plan.processing_steps) == 10
    assert len(plan.settings.variance_components.components) == 11
