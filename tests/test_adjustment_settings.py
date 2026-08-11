import pytest

from lunarops.estimation.adjustment_plan import LlrAdjustmentStage
from lunarops.estimation.adjustment_settings import (
    AdjustmentControlSettings,
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
        adjustment=AdjustmentControlSettings(
            prefit_gross_threshold_by_station_m={"sta_a": 2.0},
            convergence_threshold_by_block_m={"reflectorPosition": 1.0e-3},
        ),
    )
    assert settings.adjustment.prefit_gross_threshold_by_station_m == {"STAA": 2.0}
    report = settings.to_report_settings()
    assert set(report) == {
        "accuracy_screening",
        "adjustment",
        "initialization",
        "robust_weights",
        "variance_components",
    }
    assert "components" not in report["variance_components"]


def test_stage_and_robust_settings_validate_direct_construction():
    with pytest.raises(ValueError, match="parameter update factor"):
        LlrAdjustmentStage(name="joint", parameter_update_factor=1.1)
    with pytest.raises(ValueError, match="selectors must be unique"):
        LlrAdjustmentStage(name="joint", parametrizations=("a", " a "))
    assert RobustWeightSettings(model=" igg3 ").model == "igg3"
    with pytest.raises(ValueError, match="Active weight threshold"):
        RobustWeightSettings(active_weight_threshold=0.0)
