"""Generalized robust LLR adjustment with typed products and restart state."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Mapping, cast

from lunarops.config.context import RunContext
from lunarops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    build_processor,
    load_datasets,
    model_compatibility_fingerprint,
)
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import observation_fields, validate_adjustment_config

_ADJUSTMENT_OUTPUT_KEYS = (
    "outputFileAdjustmentReport",
    "outputFileAdjustmentState",
    "outputFileSolution",
    "outputFileCovariance",
    "outputFileNormalEquations",
)


def _scientific_fingerprint(config: Mapping[str, object], context: RunContext) -> str:
    """Hash resolved scientific settings and the contents of referenced files."""
    from lunarops.config.fingerprints import scientific_fingerprint

    excluded = {
        *_ADJUSTMENT_OUTPUT_KEYS,
        "inputFileAdjustmentState",
        "showProgress",
        "mpi",
    }
    return scientific_fingerprint(config, context, excluded_keys=excluded)


def _restore_state(state: Mapping[str, object], parametrization, processor) -> None:
    positions = state.get("reflectorPositions") or {}
    if not isinstance(positions, Mapping):
        raise ValueError("Adjustment state reflectorPositions must be a mapping.")
    processor.model_state.apply_reflector_positions_pa_m(positions)
    parameter_state = state.get("parametrization") or {}
    if not isinstance(parameter_state, Mapping):
        raise ValueError("Adjustment state parametrization must be a mapping.")
    for block in parametrization.blocks:
        saved = parameter_state.get(block.block_id)
        if not isinstance(saved, Mapping):
            continue
        values = saved.get("values")
        if values is not None:
            if not isinstance(values, Mapping) or not hasattr(block, "values"):
                raise ValueError(f"Invalid restart values for {block.block_id}.")
            block.values.update({str(key): float(value) for key, value in values.items()})


def _estimated_values(names, parametrization, processor):
    values_by_name = {}
    for block in parametrization.blocks:
        block_names = block.parameter_names()
        if block.block_id == "reflectorPosition":
            for name in block_names:
                axis = {"position.x": 0, "position.y": 1, "position.z": 2}[name.parameter_type]
                values_by_name[name] = float(
                    processor.model_state.reflector_catalog[name.object_name].moon_fixed_xyz_m[axis]
                )
        elif block.block_id == "stationRangeBias":
            keys = list(getattr(block, "keys", ()))
            for name, key in zip(block_names, keys):
                values_by_name[name] = float(block.values[key])
        elif block_names:
            raise ValueError(f"Adjustment output does not define absolute-state semantics for {block.block_id!r}.")
    return [values_by_name[name] for name in names]


@program(
    ProgramSpec(
        name="LlrAdjustment",
        summary="Run GROOPS-style nonlinear LLR processing steps with robust weighting and VCE.",
        inputs=(
            ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),
            ArtifactSlot(
                "inputFileAdjustmentState",
                "AdjustmentStateFile",
                required=False,
            ),
        ),
        outputs=(
            ArtifactSlot("outputFileAdjustmentReport", "AdjustmentReportFile"),
            ArtifactSlot("outputFileAdjustmentState", "AdjustmentStateFile"),
            ArtifactSlot("outputFileSolution", "ParameterVectorFile"),
            ArtifactSlot("outputFileCovariance", "CovarianceMatrixFile"),
            ArtifactSlot("outputFileNormalEquations", "NormalEquationFile"),
        ),
        fields=observation_fields(parametrized=True, adjustment=True),
        validator=validate_adjustment_config,
    )
)
def llr_adjustment(config: dict, context: RunContext):
    import numpy as np

    from lunarops.estimation.adjustment_config import parse_adjustment_plan
    from lunarops.estimation.uncertainty_conventions import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER
    from lunarops.estimation.parameter_products import CovarianceMatrix, ParameterVector
    from lunarops.estimation.adjustment_result_models import LlrAdjustmentResult
    from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver
    from lunarops.fileio.adjustment_artifacts import (
        read_adjustment_state,
        write_adjustment_report,
        write_adjustment_state,
    )
    from lunarops.fileio.covariance import write_covariance
    from lunarops.fileio.parameter_vectors import write_parameter_vector

    plan = parse_adjustment_plan(config)
    datasets = load_datasets(config, context)
    parametrization = build_parametrization(config, context)
    processor = build_processor(config, context)
    fingerprint = _scientific_fingerprint(config, context)

    previous_sigma_factors: dict[str, float] = {}
    previous_weight_factors: dict[Hashable, float] = {}
    observation_domain = None
    if config.get("inputFileAdjustmentState"):
        state = read_adjustment_state(context.resolve_path(config["inputFileAdjustmentState"]))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Adjustment-state fingerprint does not match the current inputs and model configuration.")
        _restore_state(state, parametrization, processor)
        sigma_state = state.get("sigmaFactors")
        weight_state = state.get("weightFactors")
        if not isinstance(sigma_state, Mapping) or not isinstance(weight_state, Mapping):
            raise TypeError("Adjustment-state sigmaFactors and weightFactors must be mappings.")
        previous_sigma_factors = {str(key): float(cast(Any, value)) for key, value in sigma_state.items()}
        previous_weight_factors = {int(cast(Any, key)): float(cast(Any, value)) for key, value in weight_state.items()}

    active_estimate = {"name": "joint"}

    def report_iteration(item):
        print(
            "[LlrAdjustment:adjustSigma0] "
            f"estimate={active_estimate['name']} "
            f"adjustmentIteration={item.adjustment_iteration} "
            f"sigmaWeightIteration={item.sigma_weight_iteration} "
            f"active={item.active_observation_count} "
            f"rejected={item.rejected_observation_count}",
            flush=True,
        )

    processing_results: list[dict[str, object]] = []
    equation_source = build_equation_source(config, context, datasets, processor)
    result: LlrAdjustmentResult | None = None
    from lunarops.estimation.adjustment_plan import EstimateStep, SelectParametrizationsStep

    available_parametrizations = tuple(block.block_id for block in parametrization.blocks)
    selected_parametrizations = available_parametrizations
    estimate_steps = [step for step in plan.processing_steps if isinstance(step, EstimateStep)]
    estimate_index = 0
    for step in plan.processing_steps:
        if isinstance(step, SelectParametrizationsStep):
            selected_parametrizations = step.apply(available_parametrizations)
            processing_results.append(
                {
                    "type": "selectParametrizations",
                    "selectedParametrizations": list(selected_parametrizations),
                }
            )
            continue
        if not selected_parametrizations:
            raise ValueError(f"Estimate step {step.name!r} has no enabled parametrizations.")
        unknown_thresholds = set(step.convergence_threshold_by_parametrization_m or {}) - set(
            selected_parametrizations
        )
        if unknown_thresholds:
            raise ValueError(
                f"Estimate step {step.name!r} has convergence thresholds for inactive or unknown "
                f"parametrizations: {sorted(unknown_thresholds)}."
            )
        estimate_index += 1
        is_final_estimate = estimate_index == len(estimate_steps)
        active_estimate["name"] = step.name
        estimate_parametrization = parametrization.select_blocks(selected_parametrizations)
        estimate_result = LlrAdjustmentSolver(
            equation_source=equation_source,
            parametrization=estimate_parametrization,
            settings=step.apply(plan.settings),
            model_state=processor.model_state,
            initial_sigma_factors=previous_sigma_factors or None,
            initial_weight_factors=previous_weight_factors or None,
            observation_domain=observation_domain,
            iteration_callback=(report_iteration if bool(config.get("showProgress", True)) else None),
        ).run(finalize=is_final_estimate)
        previous_sigma_factors = dict(estimate_result.sigma_factors)
        previous_weight_factors = {
            int(cast(Any, key)): float(value) for key, value in estimate_result.weight_factors.items()
        }
        if hasattr(estimate_result, "observation_domain"):
            observation_domain = estimate_result.observation_domain
        processing_results.append(
            {
                "type": "estimate",
                "name": step.name,
                "parametrizations": [block.block_id for block in estimate_parametrization.blocks],
                "settings": step.apply(plan.settings).to_report_settings(),
                "summary": estimate_result.summary,
                "state": estimate_result.state,
            }
        )
        if is_final_estimate:
            if not isinstance(estimate_result, LlrAdjustmentResult):
                raise RuntimeError("Final estimate step did not produce report products.")
            result = estimate_result
    if result is None:
        raise RuntimeError("Adjustment produced no final normal equations.")
    result.normals.meta["compatibility"] = model_compatibility_fingerprint(config, context)

    correction = result.remaining_correction
    cofactor = result.cofactor
    sigma0 = result.sigma0_post
    names = tuple(result.normals.parameter_names)
    units = tuple(result.normals.parameter_units)
    estimates = np.asarray(_estimated_values(names, parametrization, processor))
    cofactor_sigma = np.sqrt(np.maximum(np.diag(cofactor), 0.0))
    one_sigma = cofactor_sigma if sigma0 is None else sigma0 * cofactor_sigma
    uncertainties = PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER * one_sigma
    covariance_values = cofactor if sigma0 is None else sigma0 * sigma0 * cofactor
    covariance_kind = "cofactor" if sigma0 is None else "posteriorCovariance"
    solution = ParameterVector(
        parameter_names=names,
        values=estimates,
        units=units,
        uncertainties=uncertainties,
        uncertainty_sigma_multiplier=PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER,
    )
    covariance = CovarianceMatrix(names, covariance_values, units, covariance_kind)

    report_payload = result.to_dict()
    report_payload.update(
        {
            "fingerprint": fingerprint,
            "processingSteps": processing_results,
            "finalRemainingCorrection": {str(name): float(value) for name, value in zip(names, correction)},
        }
    )
    state_payload = {
        "fingerprint": fingerprint,
        "lastEstimate": active_estimate["name"],
        "converged": result.converged,
        "parametrization": parametrization.state(),
        "reflectorPositions": processor.model_state.reflector_positions_pa_m(),
        "sigmaFactors": result.sigma_factors,
        "weightFactors": {str(key): float(value) for key, value in result.weight_factors.items()},
    }
    write_adjustment_report(context.resolve_path(config["outputFileAdjustmentReport"]), report_payload)
    write_adjustment_state(context.resolve_path(config["outputFileAdjustmentState"]), state_payload)
    write_parameter_vector(solution, context.resolve_path(config["outputFileSolution"]))
    write_covariance(covariance, context.resolve_path(config["outputFileCovariance"]))
    from lunarops.fileio.normal_equations import write_normal_equations

    write_normal_equations(result.normals, context.resolve_path(config["outputFileNormalEquations"]))
    print(
        f"[LlrAdjustment] converged={result.converged} "
        f"adjustmentIterations={len(result.adjustment_iterations)} "
        f"sigmaWeightIterations={len(result.sigma_weight_iterations)}"
    )
    return result


__all__ = ["llr_adjustment"]
