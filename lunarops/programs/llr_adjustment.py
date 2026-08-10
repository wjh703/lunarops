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
    from lunarops.fileio.fingerprints import scientific_fingerprint

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
        summary="Run staged nonlinear LLR adjustment with robust weighting and VCE.",
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
    from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver
    from lunarops.fileio.adjustment import (
        read_adjustment_state,
        write_adjustment_report,
        write_adjustment_state,
    )
    from lunarops.fileio.parameters import (
        write_covariance,
        write_parameter_vector,
    )

    plan = parse_adjustment_plan(config)
    datasets = load_datasets(config, context)
    parametrization = build_parametrization(config, context)
    processor = build_processor(config, context)
    fingerprint = _scientific_fingerprint(config, context)

    previous_scales: dict[str, float] = {}
    previous_factors: dict[Hashable, float] = {}
    if config.get("inputFileAdjustmentState"):
        state = read_adjustment_state(context.resolve_path(config["inputFileAdjustmentState"]))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Adjustment-state fingerprint does not match the current inputs and model configuration.")
        _restore_state(state, parametrization, processor)
        scales_state = state.get("scales")
        factors_state = state.get("robustFactors")
        if not isinstance(scales_state, Mapping) or not isinstance(factors_state, Mapping):
            raise TypeError("Adjustment state scales and robustFactors must be mappings.")
        previous_scales = {str(key): float(cast(Any, value)) for key, value in scales_state.items()}
        previous_factors = {int(cast(Any, key)): float(cast(Any, value)) for key, value in factors_state.items()}

    active_stage = {"name": "joint"}

    def report_iteration(item):
        print(
            "[LlrAdjustment:HelmertVCE] "
            f"stage={active_stage['name']} "
            f"linearization={item.linearization_iteration} "
            f"stochastic={item.stochastic_iteration} "
            f"active={item.active_observation_count} "
            f"rejected={item.rejected_observation_count} "
            f"converged={item.stochastic_converged}",
            flush=True,
        )

    stage_results = []
    equation_source = build_equation_source(config, context, datasets, processor)
    result = None
    for stage_index, stage in enumerate(plan.stages):
        active_stage["name"] = stage.name
        stage_parametrization = (
            parametrization if not stage.parametrizations else parametrization.select_blocks(stage.parametrizations)
        )
        warm = stage_index == 0 and bool(config.get("inputFileAdjustmentState"))
        warm = warm or plan.warm_start_stochastic_model_across_stages
        result = LlrAdjustmentSolver(
            equation_source=equation_source,
            parametrization=stage_parametrization,
            settings=stage.apply(plan.settings),
            model_state=processor.model_state,
            initial_scales=(previous_scales if warm else None),
            initial_factors=(previous_factors if warm else None),
            iteration_callback=(report_iteration if bool(config.get("showProgress", True)) else None),
        ).run()
        previous_scales = dict(result.scales)
        previous_factors = {int(cast(Any, key)): float(value) for key, value in result.robust_factors.items()}
        stage_results.append(
            {
                "name": stage.name,
                "parametrizations": [block.block_id for block in stage_parametrization.blocks],
                "summary": result.summary,
                "state": result.state,
            }
        )
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
        kind="estimate",
        uncertainty_sigma_multiplier=PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER,
    )
    covariance = CovarianceMatrix(names, covariance_values, units, covariance_kind)

    report_payload = result.to_dict()
    report_payload.update(
        {
            "fingerprint": fingerprint,
            "processingSteps": stage_results,
            "finalRemainingCorrection": {str(name): float(value) for name, value in zip(names, correction)},
        }
    )
    state_payload = {
        "fingerprint": fingerprint,
        "lastStage": active_stage["name"],
        "converged": result.converged,
        "parametrization": parametrization.state(),
        "reflectorPositions": processor.model_state.reflector_positions_pa_m(),
        "scales": result.scales,
        "robustFactors": {str(key): float(value) for key, value in result.robust_factors.items()},
    }
    write_adjustment_report(context.resolve_path(config["outputFileAdjustmentReport"]), report_payload)
    write_adjustment_state(context.resolve_path(config["outputFileAdjustmentState"]), state_payload)
    write_parameter_vector(solution, context.resolve_path(config["outputFileSolution"]))
    write_covariance(covariance, context.resolve_path(config["outputFileCovariance"]))
    from lunarops.fileio.normal_equation_file import write_normal_equations

    write_normal_equations(result.normals, context.resolve_path(config["outputFileNormalEquations"]))
    print(
        f"[LlrAdjustment] converged={result.converged} "
        f"linearizations={len(result.linearizations)} "
        f"stochasticIterations={len(result.iterations)}"
    )
    return result


__all__ = ["llr_adjustment"]
