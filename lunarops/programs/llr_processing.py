"""GROOPS-style nonlinear LLR processing with explicit output steps."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, cast

from lunarops.config.context import RunContext
from lunarops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    build_processor,
    load_datasets,
    model_compatibility_fingerprint,
)
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import observation_fields, validate_processing_config


def _scientific_fingerprint(config: Mapping[str, object], context: RunContext) -> str:
    from lunarops.config.fingerprints import scientific_fingerprint

    scientific_steps = [
        step
        for step in cast(list[dict[str, object]], config["processingSteps"])
        if step.get("type") in {"screenObservations", "selectParametrizations", "estimate"}
    ]
    selected = {**config, "processingSteps": scientific_steps}
    return scientific_fingerprint(
        selected,
        context,
        excluded_keys={"inputFileProcessingState", "showProgress", "mpi"},
    )


def _restore_state(state: Mapping[str, object], parametrization, processor) -> None:
    positions = state.get("reflectorPositions") or {}
    if not isinstance(positions, Mapping):
        raise ValueError("Processing state reflectorPositions must be a mapping.")
    processor.model_state.apply_reflector_positions_pa_m(positions)
    parameter_state = state.get("parametrization") or {}
    if not isinstance(parameter_state, Mapping):
        raise ValueError("Processing state parametrization must be a mapping.")
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
            for name, key in zip(block_names, list(getattr(block, "keys", ()))):
                values_by_name[name] = float(block.values[key])
        elif block_names:
            raise ValueError(f"Processing output does not define absolute-state semantics for {block.block_id!r}.")
    return [values_by_name[name] for name in names]


def _result_products(result, parametrization, processor):
    import numpy as np

    from lunarops.estimation.parameter_products import CovarianceMatrix, ParameterVector
    from lunarops.estimation.uncertainty_conventions import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER

    names = tuple(result.normals.parameter_names)
    units = tuple(result.normals.parameter_units)
    estimates = np.asarray(_estimated_values(names, parametrization, processor))
    cofactor_sigma = np.sqrt(np.maximum(np.diag(result.cofactor), 0.0))
    one_sigma = cofactor_sigma if result.sigma0_post is None else result.sigma0_post * cofactor_sigma
    covariance_values = (
        result.cofactor if result.sigma0_post is None else result.sigma0_post**2 * result.cofactor
    )
    solution = ParameterVector(
        parameter_names=names,
        values=estimates,
        units=units,
        uncertainties=PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER * one_sigma,
        uncertainty_sigma_multiplier=PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER,
    )
    covariance = CovarianceMatrix(
        names,
        covariance_values,
        units,
        "cofactor" if result.sigma0_post is None else "posteriorCovariance",
    )
    return solution, covariance


def _write_residuals(step, result, datasets, context: RunContext) -> None:
    from lunarops.fileio.observation_results import write_observation_results

    source_by_identity = {
        str(record.index): source for source, dataset in datasets.items() for record in dataset.records
    }
    rows_by_source: dict[str, list[dict[str, object]]] = {source: [] for source in datasets}
    standard_fields = (
        "observation_id",
        "epoch",
        "station_id",
        "station",
        "current_state_residual_m",
        "linearized_postfit_residual_m",
        "residual_sigma_m",
        "standardized_residual",
        "applied_weight_factor",
        "applied_weight_status",
    )
    for raw in result.observations:
        row = dict(raw)
        identity = str(row["observation_id"])
        source = source_by_identity.get(identity, "processing")
        if step.output_level == "standard":
            row = {name: row[name] for name in standard_fields}
        elif isinstance(row.get("matched_parameter_names"), list):
            row["matched_parameter_names"] = ";".join(str(value) for value in row["matched_parameter_names"])
        rows_by_source.setdefault(source, []).append(row)
    rows_by_source = {source: rows for source, rows in rows_by_source.items() if rows}
    write_observation_results(rows_by_source, context.resolve_path(step.output_file))


@program(
    ProgramSpec(
        name="LlrProcessing",
        summary="Run nonlinear LLR estimation and explicit processing output steps.",
        inputs=(
            ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),
            ArtifactSlot("inputFileProcessingState", "ProcessingStateFile", required=False),
        ),
        fields=observation_fields(parametrized=True, processing=True),
        validator=validate_processing_config,
    )
)
def llr_processing(config: dict, context: RunContext):
    from lunarops.estimation.adjustment_config import parse_adjustment_plan
    from lunarops.estimation.adjustment_plan import (
        EstimateStep,
        ScreenObservationsStep,
        SelectParametrizationsStep,
        WriteNormalEquationsStep,
        WriteResidualsStep,
        WriteResultsStep,
    )
    from lunarops.estimation.adjustment_result_models import LlrAdjustmentResult
    from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver
    from lunarops.estimation.adjustment_preprocessing import screen_observations
    from lunarops.fileio.processing_artifacts import (
        read_processing_state,
        write_processing_report,
        write_processing_state,
    )

    plan = parse_adjustment_plan(config)
    datasets = load_datasets(config, context)
    parametrization = build_parametrization(config, context)
    processor = build_processor(config, context)
    fingerprint = _scientific_fingerprint(config, context)

    previous_sigma_factors: dict[str, float] = {}
    previous_weight_factors: dict[Hashable, float] = {}
    observation_domain = None
    if config.get("inputFileProcessingState"):
        state = read_processing_state(context.resolve_path(config["inputFileProcessingState"]))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Processing-state fingerprint does not match the current inputs and model configuration.")
        _restore_state(state, parametrization, processor)
        previous_sigma_factors = {
            str(key): float(cast(Any, value)) for key, value in cast(Mapping, state["sigmaFactors"]).items()
        }
        previous_weight_factors = {
            int(cast(Any, key)): float(cast(Any, value))
            for key, value in cast(Mapping, state["weightFactors"]).items()
        }

    active_estimate = {"name": "joint"}

    def report_iteration(item):
        print(
            "[LlrProcessing:estimateVarianceFactors] "
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
    available = tuple(block.block_id for block in parametrization.blocks)
    selected = available
    estimate_steps = [step for step in plan.processing_steps if isinstance(step, EstimateStep)]
    estimate_index = 0

    for step in plan.processing_steps:
        if isinstance(step, ScreenObservationsStep):
            residual_screening, reported_sigma_screening = step.screening_settings()
            observation_domain = screen_observations(
                equation_source(0),
                parametrization,
                model_state=processor.model_state,
                residual=residual_screening,
                reported_sigma=reported_sigma_screening,
                variance_components=plan.settings.variance_components,
            )
            processing_results.append(
                {
                    "type": "screenObservations",
                    "residual": {
                        "maximumAbsoluteM": step.maximum_absolute_residual_m,
                        "maximumAbsoluteByStationM": step.maximum_absolute_residual_by_station_m,
                        "rejectedCount": len(observation_domain.gross_rejected),
                    },
                    "reportedSigma": {
                        "minimumOneWayM": step.minimum_reported_one_way_sigma_m,
                        "minimumFractionOfGroupMedian": (
                            step.minimum_reported_sigma_fraction_of_group_median
                        ),
                        "rejectedCount": sum(
                            record["status"] == "REJECTED"
                            for record in observation_domain.accuracy_records.values()
                        ),
                    },
                    "retainedCount": len(observation_domain.retained_keys),
                }
            )
            continue
        if isinstance(step, SelectParametrizationsStep):
            selected = step.apply(available)
            processing_results.append({"type": "selectParametrizations", "parametrizations": list(selected)})
            continue
        if isinstance(step, EstimateStep):
            if not selected:
                raise ValueError(f"Estimate step {step.name!r} has no enabled parametrizations.")
            unknown = set(step.convergence_threshold_by_parametrization_m or {}) - set(selected)
            if unknown:
                raise ValueError(
                    f"Estimate step {step.name!r} has thresholds for inactive parametrizations: {sorted(unknown)}."
                )
            estimate_index += 1
            active_estimate["name"] = step.name
            estimate_parametrization = parametrization.select_blocks(selected)
            estimate_result = LlrAdjustmentSolver(
                equation_source=equation_source,
                parametrization=estimate_parametrization,
                settings=step.apply(plan.settings),
                model_state=processor.model_state,
                initial_sigma_factors=previous_sigma_factors or None,
                initial_weight_factors=previous_weight_factors or None,
                observation_domain=observation_domain,
                iteration_callback=report_iteration if bool(config.get("showProgress", True)) else None,
            ).run(finalize=estimate_index == len(estimate_steps))
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
            if estimate_index == len(estimate_steps):
                if not isinstance(estimate_result, LlrAdjustmentResult):
                    raise RuntimeError("Final estimate did not produce processing products.")
                result = estimate_result
                result.normals.meta["compatibility"] = model_compatibility_fingerprint(config, context)
            continue

        if result is None:
            raise RuntimeError("Output steps require a completed final estimate.")
        if isinstance(step, WriteResidualsStep):
            _write_residuals(step, result, datasets, context)
            processing_results.append({"type": "writeResiduals", "outputFile": step.output_file})
            continue
        if isinstance(step, WriteNormalEquationsStep):
            from lunarops.fileio.normal_equations import write_normal_equations

            write_normal_equations(result.normals, context.resolve_path(step.output_file))
            processing_results.append({"type": "writeNormalEquations", "outputFile": step.output_file})
            continue
        if isinstance(step, WriteResultsStep):
            from lunarops.fileio.catalogs import write_reflector_catalog
            from lunarops.fileio.covariance import write_covariance
            from lunarops.fileio.parameter_vectors import write_parameter_vector

            solution, covariance = _result_products(result, parametrization, processor)
            state_payload = {
                "fingerprint": fingerprint,
                "lastEstimate": active_estimate["name"],
                "converged": result.converged,
                "parametrization": parametrization.state(),
                "reflectorPositions": processor.model_state.reflector_positions_pa_m(),
                "sigmaFactors": result.sigma_factors,
                "weightFactors": {str(key): float(value) for key, value in result.weight_factors.items()},
            }
            processing_results.append(
                {
                    "type": "writeResults",
                    "outputFileReport": step.output_file_report,
                    "outputFileState": step.output_file_state,
                    "outputFileSolution": step.output_file_solution,
                    "outputFileCovariance": step.output_file_covariance,
                    "outputFileReflectorCatalog": step.output_file_reflector_catalog,
                }
            )
            report_payload = result.to_dict()
            report_payload.update(
                {
                    "fingerprint": fingerprint,
                    "processingSteps": processing_results,
                    "finalRemainingCorrection": {
                        str(name): float(value)
                        for name, value in zip(result.normals.parameter_names, result.remaining_correction)
                    },
                }
            )
            if step.output_file_report:
                write_processing_report(context.resolve_path(step.output_file_report), report_payload)
            if step.output_file_state:
                write_processing_state(context.resolve_path(step.output_file_state), state_payload)
            if step.output_file_solution:
                write_parameter_vector(solution, context.resolve_path(step.output_file_solution))
            if step.output_file_covariance:
                write_covariance(covariance, context.resolve_path(step.output_file_covariance))
            if step.output_file_reflector_catalog:
                write_reflector_catalog(
                    processor.model_state.reflector_catalog,
                    context.resolve_path(step.output_file_reflector_catalog),
                )
            continue
        raise TypeError(f"Unsupported processing step {type(step).__name__}.")

    if result is None:
        raise RuntimeError("LlrProcessing produced no final estimate.")
    print(
        f"[LlrProcessing] converged={result.converged} "
        f"adjustmentIterations={len(result.adjustment_iterations)} "
        f"sigmaWeightIterations={len(result.sigma_weight_iterations)}"
    )
    return result


__all__ = ["llr_processing"]
