import pytest

from lunarops.estimation.adjustment_plan import EstimateStep, SelectParametrizationsStep
from lunarops.estimation.adjustment_plan import (
    EstimateControl,
    ResidualScreening,
    LlrAdjustmentSettings,
    RobustWeightSettings,
    VarianceComponentSettings,
)
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition


def _components():
    return VarianceComponentSettings(
        (VarianceComponentDefinition("A", "STA_A", "2020-01-01", None),)
    )


def test_settings_serialize_canonical_scientific_controls():
    settings = LlrAdjustmentSettings(
        variance_components=_components(),
        residual=ResidualScreening(maximumAbsoluteByStationM={"sta_a": 2.0}),
        adjustment=EstimateControl(
            convergenceThresholdByParametrizations={"reflectorPosition": 1.0e-3},
        ),
    )
    assert settings.residual.maximumAbsoluteByStationM == {"sta_a": 2.0}
    from dataclasses import asdict
    report = {"observationScreening": {"residual": asdict(settings.residual)}, "estimate": asdict(settings.adjustment), "robustWeighting": asdict(settings.robust_weights)}
    assert set(report) == {"estimate", "observationScreening", "robustWeighting"}
    assert report["observationScreening"]["residual"]["maximumAbsoluteByStationM"] == {
        "sta_a": 2.0
    }
    assert report["estimate"]["estimateVarianceFactors"] is True


def test_processing_step_and_robust_settings_validate_direct_construction():
    with pytest.raises(ValueError, match="maxIterationCount"):
        EstimateStep(name="joint", adjustment=EstimateControl(maxIterationCount=0))
    assert RobustWeightSettings(model=" igg3 ").model == "igg3"


def test_select_parametrizations_declares_complete_set_in_available_order():
    available = ("reflectorPosition", "stationRangeBias", "earthOrientation")
    selected = SelectParametrizationsStep(
        parametrizations=("stationRangeBias", "reflectorPosition")
    ).apply(available)
    assert selected == ("reflectorPosition", "stationRangeBias")


def test_select_parametrizations_rejects_unknown_block():
    with pytest.raises(KeyError, match="Unknown parametrization"):
        SelectParametrizationsStep(parametrizations=("missing",)).apply(("reflectorPosition",))
