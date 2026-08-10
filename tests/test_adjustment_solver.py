import json
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from lunarops.classes.time import Epoch, TimeScale
from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import Parametrization, ParametrizationList
from lunarops.classes.parametrization.station_range_bias import StationRangeBiasParametrization
from lunarops.estimation.adjustment_preprocessing import floor_prefit_uncertainties
from lunarops.estimation.parameter_convergence import ParameterConvergencePolicy
from lunarops.estimation.linearization import (
    DenseLinearization,
    build_normal_equations_streaming,
)
from lunarops.estimation.normal_equation_solver import solve_normal_equations
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.estimation.adjustment_settings import (
    AdjustmentControlSettings,
    InitializationSettings,
    LlrAdjustmentSettings,
    RobustEstimationSettings,
    VarianceComponentEstimationSettings,
)
from lunarops.estimation.adjustment_solver import (
    LlrAdjustmentSolver,
)
from lunarops.estimation.robust_weights import (
    DirectRejectionWeightModel,
    Igg3WeightModel,
    active_set_change_fraction,
    direct_rejection_factors,
    igg3_factors,
    maximum_robust_factor_change,
    robust_factor_change_quantile,
)
from lunarops.estimation.helmert_vce import HelmertVceEstimator
from lunarops.estimation.variance_component_groups import (
    VarianceComponentDefinition,
    assign_variance_components,
)


def _equation(
    identity: object,
    value: float,
    station: str,
    wavelength: float = 532.0,
) -> ObservationEquation:
    return ObservationEquation(
        observed_minus_computed_one_way_m=float(value),
        sigma_one_way_m=1.0,
        design_partials={"test": np.array([1.0])},
        observation_id=identity,
        station_key=station,
        reflector_key="REF",
        transmit_epoch_utc=Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC),
        wavelength_nm=wavelength,
    )


def _settings(*, components, **overrides) -> LlrAdjustmentSettings:
    adjustment_keys = {
        "prefit_gross_threshold_m",
        "prefit_gross_threshold_by_station_m",
        "maximum_linearizations",
        "parameter_update_factor",
        "uncertainty_floor_minimum_m",
        "uncertainty_floor_group_median_fraction",
        "update_tolerance_m",
        "update_tolerance_by_block_m",
        "required_consecutive_converged_linearizations",
    }
    initialization_keys = {
        "minimum_mad_count",
        "minimum_initial_scale",
        "bias_weight_cap",
        "bias_maximum_iterations",
    }
    robust_keys = {
        "robust_model",
        "k0",
        "k1",
        "minimum_one_minus_leverage",
        "minimum_nonzero_robust_factor",
        "minimum_robust_factor_for_convergence",
        "robust_factor_change_quantile",
    }
    vce_keys = {
        "maximum_stochastic_iterations",
        "minimum_effective_redundancy",
        "scale_log_tolerance",
        "robust_factor_change_tolerance",
        "active_set_change_tolerance",
        "minimum_variance_ratio_per_iteration",
        "maximum_variance_ratio_per_iteration",
    }
    unknown = set(overrides) - adjustment_keys - initialization_keys - robust_keys - vce_keys
    if unknown:
        raise AssertionError(f"Unknown test adjustment settings: {sorted(unknown)!r}.")
    robust = {
        ("model" if key == "robust_model" else key): overrides[key]
        for key in robust_keys
        if key in overrides
    }
    robust = {
        {
            "minimum_nonzero_robust_factor": "active_factor_threshold",
            "minimum_robust_factor_for_convergence": "convergence_factor_floor",
            "robust_factor_change_quantile": "change_quantile",
        }.get(key, key): value
        for key, value in robust.items()
    }
    return LlrAdjustmentSettings(
        adjustment=AdjustmentControlSettings(
            **{key: overrides[key] for key in adjustment_keys if key in overrides}
        ),
        initialization=InitializationSettings(
            **{key: overrides[key] for key in initialization_keys if key in overrides}
        ),
        robust_estimation=RobustEstimationSettings(**robust),
        vce=VarianceComponentEstimationSettings(
            components=components,
            **{key: overrides[key] for key in vce_keys if key in overrides},
        ),
    )


class OffsetParametrization(Parametrization):
    def __init__(self):
        self.value = 0.0

    def parameter_names(self):
        return [ParameterName("test", "position.x")]

    def design_columns(self, eq):
        return np.array([1.0])

    def reduce_observation(self, eq):
        return self.value

    def apply_update(self, delta):
        self.value += float(delta[0])


class AffineParametrization(Parametrization):
    def __init__(self):
        self.value = np.zeros(2)

    def parameter_names(self):
        return [
            ParameterName("test", "position.x"),
            ParameterName("test", "position.y"),
        ]

    def design_columns(self, eq):
        station_offset = 10.0 if eq.station_key == "STA_B" else 0.0
        observation_id = cast(Any, eq.observation_id)
        return np.array([1.0, 1.0e4 * (observation_id[1] + station_offset)])

    def reduce_observation(self, eq):
        return float(self.design_columns(eq) @ self.value)

    def apply_update(self, delta):
        self.value += np.asarray(delta, dtype=float)


def test_parametrization_selection_reuses_block_state():
    offset = OffsetParametrization()
    parametrization = ParametrizationList([offset])

    selected = parametrization.select_blocks(["OffsetParametrization"])
    selected.blocks[0].apply_update(np.array([2.0]))

    assert offset.value == pytest.approx(2.0)
    with pytest.raises(KeyError, match="Unknown parametrization"):
        parametrization.select_blocks(["MissingParametrization"])


def test_parametrization_list_blocks_are_immutable():
    parametrization = ParametrizationList([OffsetParametrization()])
    dynamic_parametrization = cast(Any, parametrization)

    assert isinstance(parametrization.blocks, tuple)
    with pytest.raises(AttributeError):
        dynamic_parametrization.blocks.append(OffsetParametrization())
    with pytest.raises(AttributeError):
        dynamic_parametrization.blocks = ()


@pytest.mark.parametrize("selectors", ["OffsetParametrization", ["OffsetParametrization", "OffsetParametrization"]])
def test_parametrization_selection_rejects_invalid_selector_contract(selectors):
    parametrization = ParametrizationList([OffsetParametrization()])

    with pytest.raises((TypeError, ValueError)):
        parametrization.select_blocks(selectors)


def test_parameter_convergence_policy_supports_block_tolerances():
    policy = ParameterConvergencePolicy(
        default_tolerance_m=1.0e-3,
        tolerance_by_block_m={"stationRangeBias": 2.0e-3},
    )
    evaluation = policy.evaluate(
        {
            "reflectorPosition": 0.9e-3,
            "stationRangeBias": 1.5e-3,
        }
    )

    assert evaluation.converged
    assert evaluation.tolerances_m["reflectorPosition"] == pytest.approx(1.0e-3)
    assert evaluation.tolerances_m["stationRangeBias"] == pytest.approx(2.0e-3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_parameter_convergence_policy_rejects_invalid_update_norms(value):
    policy = ParameterConvergencePolicy(default_tolerance_m=1.0e-3)

    with pytest.raises(ValueError, match="finite and non-negative"):
        policy.evaluate({"OffsetParametrization": value})


def test_helmert_vce_requires_at_least_one_component():
    with pytest.raises(ValueError, match="at least one component"):
        HelmertVceEstimator(components=())


def test_variance_component_assignment_rejects_duplicate_observation_identities():
    component = VarianceComponentDefinition("A", "STA_A", "2010-01-01", None)

    with pytest.raises(ValueError, match="not unique"):
        assign_variance_components(
            [_equation("duplicate", 0.0, "STA_A"), _equation("duplicate", 0.0, "STA_A")],
            (component,),
        )

def test_vce_assignment_distinguishes_overlapping_cerga_systems_by_wavelength():
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "CERGA_MEO",
                "station": "GRASSE",
                "start": "2015-01-01",
                "endExclusive": "2023-01-01",
                "wavelengthMaxExclusiveNm": 700.0,
            }
        ),
        VarianceComponentDefinition.from_config(
            {
                "id": "CERGA_IR",
                "station": "GRASSE",
                "start": "2015-01-01",
                "endExclusive": None,
                "wavelengthMinNm": 700.0,
            }
        ),
    )
    equations = [
        _equation("green", 0.0, "CERGA", 532.0),
        _equation("boundary", 0.0, "CERGA", 700.0),
        _equation("infrared", 0.0, "CERGA", 1064.0),
    ]

    assert assign_variance_components(equations, components) == {
        "green": "CERGA_MEO",
        "boundary": "CERGA_IR",
        "infrared": "CERGA_IR",
    }


def test_prefit_uncertainty_qc_floors_only_abnormally_small_sigmas():
    equations = [
        replace(_equation("tiny", 0.0, "STA_A"), sigma_one_way_m=1.0e-5),
        replace(_equation("normal-1", 0.0, "STA_A"), sigma_one_way_m=0.02),
        replace(_equation("normal-2", 0.0, "STA_A"), sigma_one_way_m=0.03),
    ]

    adjusted, records, groups = floor_prefit_uncertainties(
        equations,
        {equation.observation_id: "A" for equation in equations},
        minimum_sigma_m=1.0e-3,
        minimum_group_median_fraction=0.1,
    )

    assert groups["A"] == {
        "median_reported_sigma_m": pytest.approx(0.02),
        "sigma_floor_m": pytest.approx(0.002),
        "observation_count": 3,
        "floored_count": 1,
    }
    assert [equation.sigma_one_way_m for equation in adjusted] == pytest.approx([0.002, 0.02, 0.03])
    assert records["tiny"]["status"] == "FLOORED"
    assert records["tiny"]["reported_sigma_m"] == pytest.approx(1.0e-5)
    assert records["normal-1"]["status"] == "UNCHANGED"
    assert records["tiny"]["reason"] == "BELOW_PREFIT_UNCERTAINTY_FLOOR"


def test_vce_assignment_rejects_unassigned_observation():
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )
    with pytest.raises(ValueError, match="no matching component"):
        assign_variance_components([_equation(1, 0.0, "STA_B")], components)


def test_igg3_boundaries():
    factors = igg3_factors(np.array([0.0, 1.5, 1.5001, 6.0, 6.1]), k0=1.5, k1=6.0)

    assert factors[0] == 1.0
    assert factors[1] == 1.0
    assert 0.0 < factors[2] < 1.0
    assert factors[3] == 0.0
    assert factors[4] == 0.0


def test_direct_rejection_boundaries():
    factors = direct_rejection_factors(
        np.array([-3.0001, -3.0, 0.0, 3.0, 3.0001]),
        k0=3.0,
    )

    assert factors == pytest.approx([0.0, 1.0, 1.0, 1.0, 0.0])


@pytest.mark.parametrize(
    ("old_factor", "new_factor", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.5),
        (0.5, 0.55, 0.05),
        (0.0, 0.2, 0.2),
        (0.0, 1.0, 1.0),
    ],
)
def test_robust_factor_change_is_absolute(old_factor, new_factor, expected):
    assert maximum_robust_factor_change(
        {"observation": old_factor},
        {"observation": new_factor},
        ["observation"],
    ) == pytest.approx(expected)


def test_factor_change_quantile_ignores_one_chattering_observation():
    keys = list(range(1001))
    old = {key: 0.5 for key in keys}
    target = {key: 0.501 for key in keys}
    target[keys[-1]] = 1.0

    assert robust_factor_change_quantile(
        old,
        target,
        keys,
        quantile=0.999,
    ) == pytest.approx(0.001)


def test_active_set_change_fraction_counts_membership_only():
    old = {"stable": 1.0, "removed": 0.5, "added": 0.0}
    new = {"stable": 0.2, "removed": 0.0, "added": 0.3}

    assert active_set_change_fraction(
        old,
        new,
        list(old),
        active_threshold=1.0e-12,
    ) == pytest.approx(2.0 / 3.0)


def test_igg3_update_accepts_observation_missing_from_previous_targets():
    model = Igg3WeightModel()
    update = model.update(
        {"stable": 0.0, "reentered": 0.0},
        {"stable": 1.0, "reentered": 1.0},
        {"stable": 1.0},
        ["stable", "reentered"],
    )

    assert update.target_factors == {"stable": 1.0, "reentered": 1.0}
    assert update.active_set_change_fraction == 0.0


@pytest.mark.parametrize(
    "model",
    [
        lambda: Igg3WeightModel(active_threshold=0.0),
        lambda: Igg3WeightModel(convergence_floor=1.1),
        lambda: DirectRejectionWeightModel(change_quantile=0.0),
    ],
)
def test_robust_weight_models_validate_all_convergence_controls(model):
    with pytest.raises(ValueError):
        model()


@pytest.mark.parametrize(
    "model",
    [
        lambda: Igg3WeightModel(k0=cast(Any, "1.5")),
        lambda: DirectRejectionWeightModel(k0=cast(Any, "3.0")),
    ],
)
def test_robust_weight_models_reject_string_thresholds(model):
    with pytest.raises(TypeError, match="real number"):
        model()


def test_igg3_update_preserves_factor_for_temporarily_missing_observation():
    model = Igg3WeightModel()
    update = model.update(
        {"present": 0.0},
        {"present": 0.5, "temporarily_missing": 0.25},
        {"present": 0.5, "temporarily_missing": 0.25},
        ["present"],
    )

    assert update.applied_factors == {
        "present": 1.0,
        "temporarily_missing": 0.25,
    }


def test_igg3_targets_are_applied_without_damping():
    model = Igg3WeightModel(k0=1.5, k1=6.0)
    update = model.update(
        {"full": 0.0, "downweighted": 3.0, "rejected": 7.0},
        {"full": 0.2, "downweighted": 0.8, "rejected": 1.0},
        {"full": 1.0, "downweighted": 1.0, "rejected": 1.0},
        ["full", "downweighted", "rejected"],
    )

    assert update.applied_factors == update.target_factors
    assert update.applied_factors["full"] == 1.0
    assert 0.0 < update.applied_factors["downweighted"] < 1.0
    assert update.applied_factors["rejected"] == 0.0


def test_direct_rejection_targets_are_binary_and_applied_without_damping():
    model = DirectRejectionWeightModel(k0=3.0)
    update = model.update(
        {"full": 3.0, "rejected": -3.001},
        {"full": 0.2, "rejected": 1.0},
        {"full": 1.0, "rejected": 1.0},
        ["full", "rejected"],
    )

    assert update.applied_factors == update.target_factors
    assert update.applied_factors == {"full": 1.0, "rejected": 0.0}


def test_factor_change_ignores_insignificant_boundary_crossings():
    old = {"weak": 0.0, "material": 0.0}
    new = {"weak": 1.0e-4, "material": 0.01}

    change = maximum_robust_factor_change(
        old,
        new,
        list(old),
        significance_floor=1.0e-3,
    )

    assert change == pytest.approx(0.01)
    assert (
        maximum_robust_factor_change(
            {"weak": 0.0},
            {"weak": 1.0e-4},
            ["weak"],
            significance_floor=1.0e-3,
        )
        == 0.0
    )


def test_vce_direct_update_respects_variance_ratio_limit():
    equations = [_equation(index, value, "STA_A") for index, value in enumerate([0.0, 100.0, 200.0])]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )
    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=6,
            maximum_stochastic_iterations=3,
            required_consecutive_converged_linearizations=1,
            update_tolerance_m=1.0e-6,
            k0=1.0e6,
            k1=2.0e6,
            minimum_mad_count=4,
            minimum_effective_redundancy=1.0,
            scale_log_tolerance=1.0e-6,
        ),
    ).run()

    assert result.converged
    assert result.scales["A"] == pytest.approx(100.0)
    first_component = result.iterations[0].variance_components["A"]
    assert first_component["estimated_variance"] == pytest.approx(10000.0)
    assert first_component["bounded_variance_ratio"] == pytest.approx(4.0)
    assert first_component["proposed_variance"] == pytest.approx(4.0)


def _two_component_case():
    equations = [
        replace(_equation(("A", i), value, "STA_A"), sigma_one_way_m=0.5 + 0.1 * i)
        for i, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])
    ] + [
        replace(_equation(("B", i), value, "STA_B"), sigma_one_way_m=0.8 + 0.1 * i)
        for i, value in enumerate([0.0, 2.0, 1.7, -0.2, 2.4, 0.4])
    ]
    components = tuple(
        VarianceComponentDefinition.from_config(
            {
                "id": name,
                "station": f"STA_{name}",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        )
        for name in ("A", "B")
    )
    return equations, components


def test_dense_linearization_matches_streaming_normal_equations():
    equations, components = _two_component_case()
    parametrization = ParametrizationList([AffineParametrization()])
    parametrization.setup(equations, None)
    names = parametrization.parameter_names()
    assignments = assign_variance_components(equations, components)
    scales = {"A": 1.3, "B": 0.7}
    factors = {
        equation.observation_id: 0.2 + 0.8 * (index + 1) / len(equations) for index, equation in enumerate(equations)
    }
    weights = np.asarray(
        [
            factors[equation.observation_id]
            / (scales[assignments[equation.observation_id]] ** 2 * equation.sigma_one_way_m**2)
            for equation in equations
        ]
    )

    dense = DenseLinearization.build(equations, parametrization, names)
    dense_normals = dense.normal_equations(weights)
    streaming_normals = build_normal_equations_streaming(
        equations,
        parametrization,
        parameter_names=names,
        weight_for=lambda equation: weights[equations.index(equation)],
    )
    assert np.array_equal(dense_normals.N, dense_normals.N.T)
    assert dense_normals.N == pytest.approx(streaming_normals.N, rel=1.0e-13)
    assert dense_normals.W == pytest.approx(streaming_normals.W, rel=1.0e-13)
    assert dense_normals.lPl == pytest.approx(streaming_normals.lPl, rel=1.0e-13)

    dense_solved = solve_normal_equations(dense_normals)
    streaming_solved = solve_normal_equations(streaming_normals)
    assert dense_solved.delta == pytest.approx(streaming_solved.delta, rel=1.0e-13)
    assert dense_solved.covariance == pytest.approx(streaming_solved.covariance, rel=1.0e-13)


def _run_adjustment(*, initial_scales=None, initial_factors=None):
    equations, components = _two_component_case()
    return LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=2,
            maximum_stochastic_iterations=3,
            required_consecutive_converged_linearizations=99,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
        ),
        initial_scales=initial_scales,
        initial_factors=initial_factors,
    ).run()


def test_adjustment_is_warm_startable():
    first = _run_adjustment()
    warm = _run_adjustment(initial_scales=first.scales, initial_factors=first.robust_factors)
    assert warm.settings["warm_started_scale_count"] == 2
    assert warm.settings["warm_started_factor_count"] == 12
    warm_summary = cast(dict[str, Any], warm.summary)
    assert set(cast(dict[str, float], warm_summary["performance_seconds"])) == {
        "cache_build",
        "normal_solve",
        "leverage",
        "vce",
    }


def test_llr_adjustment_runs_joint_helmert_vce_cycle():
    equations = [
        _equation(("A", index), value, "STA_A") for index, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])
    ] + [_equation(("B", index), value, "STA_B") for index, value in enumerate([0.0, 2.0, 1.7, -0.2, 2.4, 0.4])]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
        VarianceComponentDefinition.from_config(
            {
                "id": "B",
                "station": "STA_B",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )
    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=6,
            maximum_stochastic_iterations=4,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
        ),
    ).run()

    assert set(result.scales) == {"A", "B"}
    assert result.normals is not None
    assert len(result.observations) == len(equations)
    observations = cast(list[dict[str, Any]], result.observations)
    for item in observations:
        base_sigma = item["base_scale"] * item["effective_sigma_m"]
        assert 0.0 <= item["leverage"] < 1.0
        assert item["residual_sigma_m"] == pytest.approx(base_sigma * np.sqrt(1.0 - item["leverage"]))
        assert item["standardized_residual"] == pytest.approx(
            item["current_state_residual_m"] / item["residual_sigma_m"]
        )
        assert item["reported_sigma_m"] == pytest.approx(item["effective_sigma_m"])
        assert item["effective_sigma_m"] == pytest.approx(item["effective_sigma_m"])
        assert item["uncertainty_qc_status"] == "UNCHANGED"
    assert all(0.0 <= factor <= 1.0 for factor in result.robust_factors.values())
    assert result.iterations[-1].total_effective_redundancy == pytest.approx(
        result.iterations[-1].expected_total_redundancy
    )
    for iteration in result.iterations:
        variance_components = cast(dict[str, dict[str, float]], iteration.variance_components)
        expected = max(
            abs(group["proposed_variance"] / group["current_variance"] - 1.0) for group in variance_components.values()
        )
        assert iteration.maximum_variance_ratio_change == pytest.approx(expected)
    payload = cast(dict[str, Any], result.to_dict())
    json.dumps(payload)
    assert payload["summary"]["source_observation_count"] == len(equations)
    assert payload["summary"]["equation_evaluation_count"] == len(payload["equation_evaluations"])
    assert payload["summary"]["parameter_uncertainty_sigma_multiplier"] == pytest.approx(3.0)
    parameter = payload["parameters"][0]
    assert parameter["formal_uncertainty_m"] is not None
    assert parameter["formal_uncertainty_m"] == pytest.approx(
        payload["summary"]["sigma0_post"] * parameter["cofactor_uncertainty_m"]
    )
    assert "formal_sigma_m" not in parameter
    assert payload["global_residuals"]["residual_m"]["count"] == len(equations)
    assert payload["variance_components"][0]["actual_start_epoch"] is not None
    counts = ("full_weight_count", "downweighted_count", "rejected_count")
    assert (
        sum(payload["variance_components"][0][key] for key in counts)
        == payload["variance_components"][0]["observation_count"]
    )
    assert payload["iterations"][0]["candidate_update_by_block_m"]
    assert payload["iterations"][0]["variance_components"]
    assert "maximum_scale_log_target_change" in payload["iterations"][0]
    assert "robust_factor_target_change_quantile" in payload["iterations"][0]
    assert "active_set_change_fraction" in payload["iterations"][0]
    assert "target_rejected_observation_count" in payload["iterations"][0]
    assert not any("damping" in key for key in payload["settings"])


def test_direct_rejection_uses_existing_vce_path_with_binary_factors():
    equations = [_equation(index, -0.2 if index % 2 else 0.2, "STA_A") for index in range(20)]
    equations.append(_equation("outlier", 20.0, "STA_A"))
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )

    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=1,
            maximum_stochastic_iterations=2,
            required_consecutive_converged_linearizations=99,
            robust_model="directRejection",
            k0=3.0,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            scale_log_tolerance=10.0,
        ),
    ).run()

    assert result.settings["robust_estimation"]["model"] == "directRejection"
    assert set(result.robust_factors.values()) <= {0.0, 1.0}
    assert result.robust_factors["outlier"] == 0.0
    assert all(result.robust_factors[index] == 1.0 for index in range(20))
    iteration = result.iterations[-1]
    assert iteration.rejected_observation_count == 1
    assert iteration.robust_factor_summary["downweighted_count"] == 0
    assert iteration.variance_components["A"]["active_count"] == 20.0
    outlier = next(item for item in result.observations if item["observation_id"] == "outlier")
    assert outlier["applied_robust_factor"] == 0.0
    assert outlier["applied_robust_status"] == "REJECTED"


def test_adjustment_reports_prefit_uncertainty_floor():
    equations = [
        replace(_equation("tiny", 0.0, "STA_A"), sigma_one_way_m=1.0e-5),
        replace(_equation("normal-1", 1.0, "STA_A"), sigma_one_way_m=0.02),
        replace(_equation("normal-2", 2.0, "STA_A"), sigma_one_way_m=0.03),
    ]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )

    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=1,
            maximum_stochastic_iterations=1,
            required_consecutive_converged_linearizations=1,
            update_tolerance_m=10.0,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
            uncertainty_floor_minimum_m=1.0e-3,
            uncertainty_floor_group_median_fraction=0.1,
        ),
    ).run()

    records = {item["observation_id"]: item for item in cast(list[dict[str, Any]], result.observations)}
    assert result.summary["uncertainty_sigma_floored_count"] == 1
    assert result.summary["retained_uncertainty_sigma_floored_count"] == 1
    assert records["tiny"]["reported_sigma_m"] == pytest.approx(1.0e-5)
    assert records["tiny"]["effective_sigma_m"] == pytest.approx(0.002)
    assert records["tiny"]["uncertainty_qc_status"] == "FLOORED"
    assert records["normal-1"]["effective_sigma_m"] == pytest.approx(0.02)
    quality_control = cast(dict[str, Any], result.uncertainty_quality_control)
    assert quality_control["groups"]["A"]["floored_count"] == 1


def test_open_bias_interval_remains_active():
    equation = _equation(1, 0.0, "WETTZELL")
    block = StationRangeBiasParametrization(
        per="station+interval",
        intervals=[
            {
                "station": "WETTZELL",
                "start": "2018-01-01",
                "end_exclusive": None,
                "name": "WETTZELL_PRESENT",
            }
        ],
    )
    block.setup([equation], None)

    assert block.keys == ["WETTZELL_PRESENT"]
    assert np.allclose(block.design_columns(equation), [1.0])


def test_prefit_gross_rejection_never_reenters():
    equations = [_equation("gross", 100.0, "STA"), _equation("good", 1.0, "STA")]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "STA",
                "station": "STA",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )
    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            components=components,
            maximum_linearizations=2,
            prefit_gross_threshold_m=20.0,
            maximum_stochastic_iterations=1,
            required_consecutive_converged_linearizations=1,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
        ),
    ).run()

    assert result.gross_rejected == {"gross": 100.0}
    assert set(result.robust_factors) == {"good"}


def test_extended_variance_components_cover_mcdonald_1969_and_current_meo():
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "MCDONALD_1969_1985",
                "station": "MCDONALD",
                "start": "1969-01-01",
                "endExclusive": "1986-01-01",
            }
        ),
        VarianceComponentDefinition.from_config(
            {
                "id": "CERGA_MEO_2009_PRESENT",
                "station": "GRASSE",
                "start": "2009-01-01",
                "endExclusive": None,
                "wavelengthMaxExclusiveNm": 700.0,
            }
        ),
    )
    mcdonald = replace(
        _equation("mcdonald-1969", 0.0, "MCDONALD", 694.3),
        transmit_epoch_utc=Epoch.from_isot("1969-08-20T00:00:00", scale=TimeScale.UTC),
    )
    meo = replace(
        _equation("meo-2024", 0.0, "GRASSE", 532.1),
        transmit_epoch_utc=Epoch.from_isot("2024-12-21T00:00:00", scale=TimeScale.UTC),
    )

    assert assign_variance_components([mcdonald, meo], components) == {
        "mcdonald-1969": "MCDONALD_1969_1985",
        "meo-2024": "CERGA_MEO_2009_PRESENT",
    }


def test_stochastic_iterations_do_not_recompute_observation_equations():
    equations = [_equation(index, value, "STA_A") for index, value in enumerate([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )
    source_calls = []

    def source(iteration):
        source_calls.append(iteration)
        return equations

    result = LlrAdjustmentSolver(
        equation_source=source,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=2,
            maximum_stochastic_iterations=5,
            required_consecutive_converged_linearizations=1,
            update_tolerance_m=10.0,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
            scale_log_tolerance=1.0e-12,
            robust_factor_change_tolerance=1.0e-12,
        ),
    ).run()

    assert len(result.iterations) > 1
    assert source_calls == [1, 2]
    assert result.converged
    assert result.termination_reason == "CONVERGED"
    assert result.summary["equation_evaluation_count"] == 2
    assert result.equation_evaluations[0]["fixed_domain_returned_count"] == 6
    assert [item["purpose"] for item in result.equation_evaluations] == [
        "initialization",
        "final-state-report",
    ]


def test_stochastic_iteration_limit_still_applies_parameter_update():
    equations = [_equation(index, value, "STA_A") for index, value in enumerate([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )
    source_calls = []

    def source(iteration):
        source_calls.append(iteration)
        return equations

    block = OffsetParametrization()
    result = LlrAdjustmentSolver(
        equation_source=source,
        parametrization=ParametrizationList([block]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=2,
            parameter_update_factor=0.5,
            maximum_stochastic_iterations=1,
            update_tolerance_m=0.0,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
            scale_log_tolerance=0.0,
            robust_factor_change_tolerance=0.0,
            active_set_change_tolerance=0.0,
        ),
    ).run()

    assert source_calls == [1, 2, 3]
    first = cast(dict[str, Any], result.linearizations[0])
    assert not first["stochastic_converged"]
    assert first["stochastic_iteration_limit_reached"]
    assert first["parameter_update_factor"] == 0.5
    candidate = first["candidate_update_by_block_m"]["OffsetParametrization"]
    assert first["maximum_parameter_update_m"] == pytest.approx(candidate)
    applied = first["applied_update_by_block_m"]["OffsetParametrization"]
    assert applied == pytest.approx(0.5 * candidate)
    assert applied > 0.0
    assert result.settings["adjustment"]["parameter_update_factor"] == 0.5
    assert result.termination_reason == "MAXIMUM_LINEARIZATIONS_REACHED"


def test_vce_settings_requires_at_least_one_variance_component():
    with pytest.raises(ValueError, match="At least one variance component"):
        VarianceComponentEstimationSettings(components=())


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.01])
def test_parameter_update_factor_must_be_in_unit_interval(factor):
    with pytest.raises(ValueError, match="Parameter update factor"):
        AdjustmentControlSettings(parameter_update_factor=factor)


def test_fixed_domain_observation_can_reenter_after_one_failed_linearization():
    equations = [_equation(index, value, "STA_A") for index, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )

    def source(iteration):
        return [
            replace(equation, light_time_converged=not (iteration == 2 and index == 0))
            for index, equation in enumerate(equations)
        ]

    result = LlrAdjustmentSolver(
        equation_source=source,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=3,
            maximum_stochastic_iterations=1,
            required_consecutive_converged_linearizations=99,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
        ),
    ).run()

    assert [item["fixed_domain_returned_count"] for item in result.equation_evaluations] == [6, 5, 6, 6]
    assert set(result.robust_factors) == set(range(6))


def test_parameter_convergence_requires_two_confirmation_linearizations():
    equations = [_equation(index, value, "STA_A") for index, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])]
    late = _equation("late", 100.0, "STA_A")
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )
    source_calls = []

    def source(iteration):
        source_calls.append(iteration)
        return equations + [replace(late, light_time_converged=iteration > 1)]

    block = OffsetParametrization()
    result = LlrAdjustmentSolver(
        equation_source=source,
        parametrization=ParametrizationList([block]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=4,
            maximum_stochastic_iterations=2,
            update_tolerance_m=1.0e-6,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            scale_log_tolerance=10.0,
            robust_factor_change_tolerance=10.0,
        ),
    ).run()

    assert source_calls == [1, 2, 3, 4]
    assert len(result.linearizations) == 3
    first_linearization = cast(dict[str, Any], result.linearizations[0])
    assert not first_linearization["parameter_converged"]
    assert first_linearization["applied_update_by_block_m"]["OffsetParametrization"] == pytest.approx(
        first_linearization["candidate_update_by_block_m"]["OffsetParametrization"]
    )
    assert "geometry_damping" not in result.settings
    assert result.summary["equation_evaluation_count"] == 4
    assert result.equation_evaluations[1]["fixed_domain_returned_count"] == 6
    assert result.linearizations[-2]["consecutive_converged_linearizations"] == 1
    assert result.linearizations[-1]["consecutive_converged_linearizations"] == 2
    assert result.converged
    assert block.value == pytest.approx(np.mean([0.7, 1.0, 1.2, 0.8, 1.1, 0.9]))
    assert "late" not in result.robust_factors
    assert all(item["observation_id"] != "late" for item in result.observations)


def test_final_report_matches_the_applied_damped_state():
    values = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    equations = [_equation(index, value, "STA_A") for index, value in enumerate(values)]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station": "STA_A",
                "start": "2010-01-01",
                "endExclusive": None,
            }
        ),
    )
    block = OffsetParametrization()

    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([block]),
        settings=_settings(
            components=components,
            prefit_gross_threshold_m=None,
            maximum_linearizations=1,
            parameter_update_factor=0.5,
            maximum_stochastic_iterations=1,
            required_consecutive_converged_linearizations=99,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            scale_log_tolerance=10.0,
            robust_factor_change_tolerance=10.0,
            k0=1.0e6,
            k1=2.0e6,
        ),
    ).run()

    assert block.value == pytest.approx(1.0)
    assert result.parameters[0]["remaining_linearized_correction_m"] == pytest.approx(1.0)
    assert result.remaining_correction == pytest.approx([1.0])
    assert not result.remaining_correction.flags.writeable
    assert not result.cofactor.flags.writeable
    assert result.sigma0_post == pytest.approx(result.summary["sigma0_post"])
    first = result.observations[0]
    assert first["current_state_residual_m"] == pytest.approx(0.0)
    assert first["linearized_postfit_residual_m"] == pytest.approx(-1.0)
    assert first["applied_robust_factor"] == pytest.approx(1.0)
    assert first["final_state_proposed_robust_factor"] == pytest.approx(1.0)
    assert not first["proposed_robust_factor_applied"]
    assert "applied_igg3_factor" not in first
    assert not result.variance_components[0]["proposed_scale_applied"]
    assert result.equation_evaluations[-1]["purpose"] == "final-state-report"


def test_helmert_vce_handles_zero_effective_redundancy_without_dividing_by_zero():
    component = VarianceComponentDefinition("A", "STA_A", "2010-01-01", None)
    normals = NormalEquations.zeros([ParameterName("test", "position.x")])
    normals.accumulate(np.array([[1.0]]), np.array([0.0]), np.array([1.0]))

    estimate = HelmertVceEstimator(
        (component,),
        minimum_effective_redundancy=0.0,
    ).estimate(
        design=np.array([[1.0]]),
        sigmas=np.array([1.0]),
        residuals=np.array([0.0]),
        component_ids=np.array(["A"], dtype=object),
        factors=np.array([1.0]),
        scales={"A": 1.0},
        normals=normals,
        covariance=np.array([[1.0]]),
    )

    assert estimate.scales == {"A": 1.0}
    assert estimate.diagnostics["A"]["update_status"] == "ZERO_EFFECTIVE_REDUNDANCY"


def test_helmert_vce_does_not_collapse_scale_for_zero_variance_target():
    component = VarianceComponentDefinition("A", "STA_A", "2010-01-01", None)
    normals = NormalEquations.zeros([ParameterName("test", "position.x")])
    normals.accumulate(np.array([[1.0], [1.0]]), np.array([0.0, 0.0]), np.ones(2))

    estimate = HelmertVceEstimator(
        (component,),
        minimum_effective_redundancy=1.0,
    ).estimate(
        design=np.array([[1.0], [1.0]]),
        sigmas=np.ones(2),
        residuals=np.zeros(2),
        component_ids=np.array(["A", "A"], dtype=object),
        factors=np.ones(2),
        scales={"A": 1.0},
        normals=normals,
        covariance=np.array([[0.5]]),
    )

    diagnostics = estimate.diagnostics["A"]
    assert estimate.scales == {"A": 1.0}
    assert diagnostics["estimated_variance"] == 0.0
    assert diagnostics["estimated_variance_ratio"] == 0.0
    assert diagnostics["bounded_variance_ratio"] == 1.0
    assert diagnostics["update_status"] == "ZERO_VARIANCE_TARGET"


def test_standardized_residuals_use_current_robust_weights_for_leverage():
    equations = [_equation(index, value, "STA_A") for index, value in enumerate([0.0, 0.0, 1.0])]
    component = VarianceComponentDefinition("A", "STA_A", "2010-01-01", None)
    parametrization = ParametrizationList([OffsetParametrization()])
    parametrization.setup(equations, None)
    solver = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=parametrization,
        settings=_settings(components=(component,), minimum_effective_redundancy=1.0),
    )
    solver._names = parametrization.parameter_names()
    solver._assignments = assign_variance_components(equations, (component,))
    solver._prepare_linearization(equations)
    factors = {0: 1.0, 1: 1.0, 2: 0.01}
    solution = solver._solve_linearized(equations, {"A": 1.0}, factors)
    _, residual_sigmas = solver._standardized_residuals(solution, {"A": 1.0})

    normal = 2.01
    assert residual_sigmas[0] == pytest.approx(np.sqrt(1.0 - 1.0 / normal))
    assert residual_sigmas[1] == pytest.approx(np.sqrt(1.0 - 1.0 / normal))
    assert residual_sigmas[2] == pytest.approx(np.sqrt(1.0 - 0.01 / normal))
    assert residual_sigmas[2] > residual_sigmas[0]


def test_solver_zeros_below_threshold_factors_consistently():
    equations = [_equation(index, value, "STA_A") for index, value in enumerate([0.0, 0.0, 1000.0])]
    component = VarianceComponentDefinition("A", "STA_A", "2010-01-01", None)
    parametrization = ParametrizationList([OffsetParametrization()])
    parametrization.setup(equations, None)
    solver = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=parametrization,
        settings=_settings(
            components=(component,),
            minimum_nonzero_robust_factor=1.0e-3,
            minimum_effective_redundancy=1.0,
        ),
    )
    solver._names = parametrization.parameter_names()
    solver._assignments = assign_variance_components(equations, (component,))
    solver._prepare_linearization(equations)

    solution = solver._solve_linearized(
        equations,
        {"A": 1.0},
        {0: 1.0, 1: 1.0, 2: 1.0e-4},
    )

    assert solution.normals.obs_count == 2
    assert solution.weights == pytest.approx([1.0, 1.0, 0.0])
    assert solution.wrms_m == pytest.approx(0.0)


def test_adjustment_fails_explicitly_when_no_light_time_solution_is_usable():
    equations = [replace(_equation(index, 0.0, "STA_A"), light_time_converged=False) for index in range(2)]
    component = VarianceComponentDefinition("A", "STA_A", "2010-01-01", None)

    with pytest.raises(ValueError, match="no light-time-converged observations"):
        LlrAdjustmentSolver(
            equation_source=lambda iteration: equations,
            parametrization=ParametrizationList([OffsetParametrization()]),
            settings=_settings(components=(component,)),
        ).run()


def test_adjustment_fails_explicitly_when_prefit_qc_rejects_every_observation():
    equations = [_equation(index, 10.0 + index, "STA_A") for index in range(2)]
    component = VarianceComponentDefinition("A", "STA_A", "2010-01-01", None)

    with pytest.raises(ValueError, match="no observations after prefit gross rejection"):
        LlrAdjustmentSolver(
            equation_source=lambda iteration: equations,
            parametrization=ParametrizationList([OffsetParametrization()]),
            settings=_settings(
                components=(component,),
                prefit_gross_threshold_m=1.0,
            ),
        ).run()


def test_adjustment_rejects_invalid_warm_start_stochastic_model_values():
    equations = [_equation(0, 0.0, "STA_A"), _equation(1, 0.0, "STA_A")]
    component = VarianceComponentDefinition("A", "STA_A", "2010-01-01", None)
    settings = _settings(components=(component,), prefit_gross_threshold_m=None)

    with pytest.raises(ValueError, match="unknown components"):
        LlrAdjustmentSolver(
            equation_source=lambda iteration: equations,
            parametrization=ParametrizationList([OffsetParametrization()]),
            settings=settings,
            initial_scales={"unknown": 1.0},
        )

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        LlrAdjustmentSolver(
            equation_source=lambda iteration: equations,
            parametrization=ParametrizationList([OffsetParametrization()]),
            settings=settings,
            initial_factors={0: 1.5},
        ).run()
