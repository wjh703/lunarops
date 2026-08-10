import pytest

from lunarops.estimation.adjustment_plan import LlrAdjustmentStage
from lunarops.estimation.adjustment_settings import (
    AdjustmentControlSettings,
    LlrAdjustmentSettings,
    RobustEstimationSettings,
    VarianceComponentEstimationSettings,
)
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition


def _vce_settings() -> VarianceComponentEstimationSettings:
    return VarianceComponentEstimationSettings(
        components=(VarianceComponentDefinition("A", "STA_A", "2020-01-01", None),),
    )


def test_adjustment_settings_group_and_serialize_the_scientific_controls():
    settings = LlrAdjustmentSettings(
        vce=_vce_settings(),
        adjustment=AdjustmentControlSettings(
            prefit_gross_threshold_by_station_m={"sta_a": 2.0},
            update_tolerance_by_block_m={"reflectorPosition": 1.0e-3},
        ),
    )

    assert settings.adjustment.prefit_gross_threshold_by_station_m == {"STAA": 2.0}
    report = settings.to_report_settings()
    assert set(report) == {"adjustment", "initialization", "robust_estimation", "vce"}
    assert "components" not in report["vce"]
    assert report["adjustment"]["update_tolerance_by_block_m"] == {"reflectorPosition": 1.0e-3}


def test_stage_model_validates_direct_construction():
    with pytest.raises(ValueError, match="Stage parameter update factor"):
        LlrAdjustmentStage(name="joint", parameter_update_factor=1.1)

    with pytest.raises(ValueError, match="selectors must be unique"):
        LlrAdjustmentStage(name="joint", parametrizations=("reflectorPosition", "reflectorPosition"))

    with pytest.raises(ValueError, match="selectors must be unique"):
        LlrAdjustmentStage(name="joint", parametrizations=("reflectorPosition", " reflectorPosition "))


def test_direct_settings_normalize_robust_model_and_reject_invalid_controls():
    assert RobustEstimationSettings(model=" igg3 ").model == "igg3"

    with pytest.raises(ValueError, match="Active robust-factor threshold"):
        RobustEstimationSettings(active_factor_threshold=0.0)


def test_direct_settings_reject_duplicate_block_tolerances_after_trimming():
    with pytest.raises(ValueError, match="unique after trimming"):
        AdjustmentControlSettings(update_tolerance_by_block_m={"reflectorPosition": 1.0e-3, " reflectorPosition ": 2.0e-3})
