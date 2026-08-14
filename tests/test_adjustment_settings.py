import pytest

from lunarops.estimation.adjustment_plan import EstimateStep, SelectParametrizationsStep
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
            convergence_threshold_by_parametrization_m={"reflectorPosition": 1.0e-3},
        ),
    )
    assert settings.adjustment.prefit_gross_threshold_by_station_m == {"STAA": 2.0}
    report = settings.to_report_settings()
    assert set(report) == {"estimate", "observation_screening", "robust_weighting"}
    assert report["observation_screening"]["residual"]["maximum_absolute_by_station_m"] == {
        "STAA": 2.0
    }
    assert report["estimate"]["estimate_variance_factors"] is True


def test_processing_step_and_robust_settings_validate_direct_construction():
    with pytest.raises(ValueError, match="max_iteration_count"):
        EstimateStep(name="joint", max_iteration_count=0)
    with pytest.raises(ValueError, match="must be unique"):
        SelectParametrizationsStep(parametrizations=("a", " a "))
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
