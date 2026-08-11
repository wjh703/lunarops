from copy import deepcopy
from pathlib import Path

import pytest

from lunarops.config.loader import load_config_file
from lunarops.estimation.adjustment_config import parse_adjustment_plan


def _component(component_id="A"):
    return {"id": component_id, "station": "STA_A", "start": "2020-01-01", "endExclusive": None}


def _config():
    return {"varianceComponents": {"components": [_component()]}}


def test_canonical_schema_maps_to_typed_plan():
    config = _config()
    config.update(
        {
            "adjustment": {
                "maxIterationCount": 9,
                "convergenceThreshold": 0.003,
                "convergenceThresholdByBlock": {"offset": 0.004},
                "stages": [{"name": "offset", "maxIterationCount": 4, "convergenceThreshold": 0.001}],
            },
            "accuracyScreening": {"minimumOneWayM": 0.002, "minimumFractionOfGroupMedian": 0.1},
            "initialization": {"biasWeightCap": 1.0e10, "biasMaximumIterations": 12},
            "robustWeights": {"model": "igg3", "k0": 1.2, "k1": 5.0},
        }
    )
    plan = parse_adjustment_plan(config)
    assert plan.settings.adjustment.max_iteration_count == 9
    assert plan.settings.accuracy_screening.minimum_one_way_m == pytest.approx(0.002)
    assert plan.settings.robust_weights.k1 == pytest.approx(5.0)
    assert plan.stages[0].apply(plan.settings).adjustment.max_iteration_count == 4


def test_direct_rejection_uses_k0_only():
    config = _config()
    config["robustWeights"] = {"model": "directRejection", "k0": 3.0}
    settings = parse_adjustment_plan(config).settings
    assert settings.robust_weights.k1 is None
    config["robustWeights"]["k1"] = 6.0
    with pytest.raises(ValueError, match="uses k0 only"):
        parse_adjustment_plan(config)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("adjustment", "maximumLinearizations"),
        ("adjustment", "uncertaintyFloor"),
        ("adjustment", "parameterUpdateFactor"),
        ("adjustment", "requiredConsecutiveConvergedIterations"),
        ("adjustment", "warmStartSigmaAndWeightsAcrossStages"),
        ("initialization", "minimumMadCount"),
        ("robustWeights", "activeWeightThreshold"),
        ("robustWeights", "minimumOneMinusLeverage"),
        ("varianceComponents", "minimumRedundancy"),
        ("varianceComponents", "minimumVarianceRatio"),
        ("varianceComponents", "maximumVarianceRatio"),
    ],
)
def test_removed_controls_are_rejected(section, key):
    config = _config()
    config.setdefault(section, {})[key] = 1
    with pytest.raises(ValueError, match="unknown key"):
        parse_adjustment_plan(config)


@pytest.mark.parametrize("section", ["vce", "robustEstimation"])
def test_old_section_names_are_not_accepted(section):
    config = _config()
    config[section] = {"components": [_component()]} if section == "vce" else {"model": "igg3"}
    with pytest.raises(ValueError, match="Obsolete adjustment section"):
        parse_adjustment_plan(config)


def test_component_schema_and_duplicate_stage_names_are_strict():
    config = _config()
    config["varianceComponents"]["components"][0]["station_system"] = "A"
    with pytest.raises(ValueError, match="unknown key"):
        parse_adjustment_plan(config)
    config = _config()
    config["adjustment"] = {"stages": [{"name": "joint"}, {"name": "joint"}]}
    with pytest.raises(ValueError, match="names must be unique"):
        parse_adjustment_plan(config)


def test_detailed_config_uses_canonical_schema():
    root = Path(__file__).resolve().parents[1]
    config = load_config_file(root / "configs" / "lunarops_reflector_bias_adjustment_detailed.yml")
    program = next(item for item in config["programs"] if item.get("program") == "LlrAdjustment")
    plan = parse_adjustment_plan(deepcopy(program))
    assert [stage.name for stage in plan.stages] == ["reflector", "bias", "joint"]
    assert len(plan.settings.variance_components.components) == 11
