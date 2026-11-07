"""processingSteps and varianceComponents resolve into a typed adjustment plan."""

import pytest

from lunarops.config.processing import PROCESSING_SCHEMA
from lunarops.estimation.adjustment_plan import SelectParametrizationsStep, parse_adjustment_plan

COMPONENT = {"id": "A", "station": "STA_A", "start": "2020-01-01", "endExclusive": None}
SCREEN = {"type": "screenObservations"}
WRITE = {"type": "writeResults", "outputFileReport": "report.txt"}


def _plan(*steps, components=(COMPONENT,)):
    return parse_adjustment_plan(PROCESSING_SCHEMA.resolve({"varianceComponents": list(components),
                                                            "processingSteps": list(steps)}))


def test_configuration_maps_onto_typed_steps_and_settings():
    plan = _plan(
        {"type": "screenObservations", "residual": {"maximumAbsoluteM": 3.0},
         "reportedSigma": {"minimumOneWayM": 0.002, "minimumFractionOfGroupMedian": 0.1}},
        {"type": "selectParametrizations", "parametrizations": ["offset"]},
        {"type": "estimate", "name": "offset", "maxIterationCount": 4, "convergenceThreshold": 0.001,
         "convergenceThresholdByParametrizations": {"offset": 0.004}, "estimateVarianceFactors": False,
         "robustWeighting": {"model": "igg3", "k0": 1.2, "k1": 5.0}},
        WRITE,
    )
    settings = plan.processing_steps[2].apply(plan)

    assert plan.processing_steps[0].accuracy.minimumOneWayM == 0.002
    assert plan.processing_steps[2].robust_weighting.k1 == 5.0
    assert settings.adjustment.maxIterationCount == 4
    assert settings.adjustment.convergenceThresholdByParametrizations == {"offset": 0.004}
    assert not settings.adjustment.estimateVarianceFactors and settings.adjustment.estimateRobustWeights
    assert settings.residual.maximumAbsoluteM == 3.0


def test_robust_weighting_defaults_and_the_direct_rejection_model():
    default = _plan(SCREEN, {"type": "estimate", "name": "joint"}, WRITE).processing_steps[1].robust_weighting
    direct = _plan(SCREEN, {"type": "estimate", "name": "joint",
                            "robustWeighting": {"model": "directRejection", "k0": 3.0}}, WRITE)

    assert (default.model, default.k0, default.k1) == ("igg3", 1.5, 6.0)
    assert direct.processing_steps[1].robust_weighting.k1 is None
    with pytest.raises(ValueError, match="uses k0 only"):
        _plan(SCREEN, {"type": "estimate", "name": "joint",
                       "robustWeighting": {"model": "directRejection", "k0": 3.0, "k1": 6.0}}, WRITE)


@pytest.mark.parametrize(("steps", "components", "message"), [
    ((SCREEN, {"type": "estimate", "name": "joint"}, WRITE), ({**COMPONENT, "station_system": "A"},),
     "unknown configuration key"),
    ((SCREEN, {"type": "estimate", "name": "joint"}, {"type": "estimate", "name": "joint"}, WRITE), (COMPONENT,),
     "names must be unique"),
    ((SCREEN, {"type": "estimate", "name": "joint", "computeResiduals": False}, WRITE), (COMPONENT,),
     r"processingSteps\[1\].*require computeResiduals=true"),
    ((SCREEN, {"type": "estimate", "name": "joint", "varianceEstimation": "legacy", "sessionCorrelation": True},
      WRITE), (COMPONENT,), r"processingSteps\[1\].*sessionCorrelation requires"),
    ((SCREEN, {"type": "selectParametrizations", "enable": ["offset"]}, {"type": "estimate", "name": "j"}, WRITE),
     (COMPONENT,), "unknown configuration key"),
    (({"type": "estimate", "name": "joint"}, SCREEN, WRITE), (COMPONENT,), "must start with exactly one"),
    ((SCREEN, SCREEN, {"type": "estimate", "name": "joint"}, WRITE), (COMPONENT,), "must start with exactly one"),
], ids=["component-key", "duplicate-estimate", "weights-without-residuals", "sessions-need-consistent",
        "obsolete-selector", "screen-not-first",
        "two-screens"])
def test_invalid_plans_are_rejected(steps, components, message):
    with pytest.raises(ValueError, match=message):
        _plan(*steps, components=components)


def test_parametrization_selection_keeps_the_available_order_and_rejects_unknown_blocks():
    available = ("reflectorPosition", "stationRangeBias", "earthOrientation")
    step = SelectParametrizationsStep(parametrizations=("stationRangeBias", "reflectorPosition"))

    assert step.apply(available) == ("reflectorPosition", "stationRangeBias")
    with pytest.raises(KeyError, match="Unknown parametrization"):
        SelectParametrizationsStep(parametrizations=("missing",)).apply(available)


def test_output_steps_follow_the_estimates():
    plan = _plan(SCREEN, {"type": "estimate", "name": "joint"},
                 {"type": "writeResiduals", "outputFile": "residuals.txt"},
                 {"type": "writeNormalEquations", "outputFile": "normals"}, WRITE)
    residuals, normals, results = plan.processing_steps[2:]

    assert (residuals.output_file, residuals.output_level) == ("residuals.txt", "standard")
    assert normals.output_file == "normals"
    assert results.outputFileReport == "report.txt" and results.outputFileSolution is None
    with pytest.raises(ValueError, match="must precede all output steps"):
        _plan(SCREEN, WRITE, {"type": "estimate", "name": "joint"})
