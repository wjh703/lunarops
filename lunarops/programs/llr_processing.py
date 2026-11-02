"""GROOPS-style nonlinear LLR processing with explicit output steps."""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from lunarops.classes.observation.assembly import build_observation_processor
from lunarops.config.context import RunContext
from lunarops.config.fingerprints import model_compatibility_fingerprint, processing_fingerprint
from lunarops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    load_datasets,
)
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import observation_fields, validate_processing_config

if TYPE_CHECKING:
    from lunarops.classes.observation import LlrObservationProcessor
    from lunarops.classes.parametrization.base import ParametrizationList
    from lunarops.estimation.adjustment_plan import LlrAdjustmentPlan
    from lunarops.llr_workflow import EquationSource


@dataclass(frozen=True, slots=True)
class ProcessingProblem:
    plan: LlrAdjustmentPlan
    datasets: dict
    parametrization: ParametrizationList
    processor: LlrObservationProcessor
    equation_source: EquationSource
    fingerprint: str
    manifest: dict[str, object]


def build_processing_problem(config, context) -> ProcessingProblem:
    from lunarops.config.fingerprints import scientific_input_manifest

    manifest = scientific_input_manifest(
        {key: value for key, value in config.items() if key != "inputFileProcessingState"},
        context,
    )
    from lunarops.estimation.adjustment_plan import parse_adjustment_plan

    plan = parse_adjustment_plan(config)
    datasets = load_datasets(config, context)
    processor = build_observation_processor(context, config)
    parametrization = build_parametrization(config, context, processor.model_state)
    return ProcessingProblem(
        plan,
        datasets,
        parametrization,
        processor,
        build_equation_source(config, context, datasets, processor),
        processing_fingerprint(config, context, plan=plan, manifest=manifest),
        manifest,
    )


def _restore_state(state: Mapping[str, object], parametrization, processor) -> None:
    positions = state["reflectorPositions"]
    processor.model_state.apply_reflector_positions_pa_m(positions)
    station_states = state.get("stationStates")
    if station_states:
        processor.model_state.apply_station_states_itrf_m(station_states)
    parameter_state = cast(Mapping, state["parametrization"])
    for block in parametrization.blocks:
        saved = parameter_state.get(block.block_id)
        if saved is None:
            continue
        values = saved.get("values")
        if values is not None:
            block.values.update({str(key): float(value) for key, value in values.items()})


def _write_residuals(step, result, datasets, context: RunContext) -> None:
    from lunarops.estimation.adjustment_reporting import OBSERVATION_REPORT_FIELDS
    from lunarops.fileio.observation_results import write_observation_results

    source_by_identity = {
        str(record.index): source for source, dataset in datasets.items() for record in dataset.records
    }
    rows_by_source: dict[str, list[dict[str, object]]] = {source: [] for source in datasets}
    standard_fields = (
        "observation_id",
        "epoch_utc",
        "epoch_local",
        "station_id",
        "station",
        "current_state_residual_m",
        "linearized_postfit_residual_m",
        "residual_sigma_m",
        "standardized_residual",
        "applied_weight_factor",
        "applied_weight_status",
    )
    columns = {name: re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
               for name, _, _ in OBSERVATION_REPORT_FIELDS}
    for raw in result.observations:
        row = {columns[key]: value for key, value in raw.items()}
        identity = str(row["observation_id"])
        source = source_by_identity.get(identity, "processing")
        if step.output_level == "standard":
            row = {name: row[name] for name in standard_fields}
        else:
            row["matched_parameter_names"] = ";".join(str(value) for value in row["matched_parameter_names"])
        rows_by_source.setdefault(source, []).append(row)
    rows_by_source = {source: rows for source, rows in rows_by_source.items() if rows}
    write_observation_results(rows_by_source, context.resolve_path(step.output_file))


@program(
    ProgramSpec(
        name="LlrProcessing",
        summary="Run nonlinear LLR estimation and explicit processing output steps.",
        inputs=(
            ArtifactSlot("inputFilesNormalPoints", "normalPoint", many=True),
            ArtifactSlot("inputFileProcessingState", "processingState", required=False),
            ArtifactSlot("inputFileStationCatalog", "stationCatalog"),
            ArtifactSlot("inputFileReflectorCatalog", "reflectorCatalog"),
        ),
        fields=observation_fields(parametrized=True, processing=True),
        validator=validate_processing_config,
    )
)
def llr_processing(config: dict, context: RunContext):
    from lunarops.classes.frames.coordinate_scaling import GM_BY_BODY
    from lunarops.estimation.adjustment_plan import (
        EstimateStep,
        ScreenObservationsStep,
        SelectParametrizationsStep,
        WriteNormalEquationsStep,
        WriteResidualsStep,
        WriteResultsStep,
    )
    from lunarops.estimation.adjustment_preprocessing import screen_observations
    from lunarops.estimation.adjustment_result_models import (
        LlrAdjustmentObservationDomain,
        LlrAdjustmentResult,
    )
    from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver
    from lunarops.fileio.processing_artifacts import (
        read_processing_state,
        write_processing_state,
    )
    from lunarops.fileio.structured_reports import write_processing_report

    problem = build_processing_problem(config, context)
    plan, datasets = problem.plan, problem.datasets
    parametrization, processor = problem.parametrization, problem.processor
    fingerprint = problem.fingerprint

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
            int(cast(Any, key)): float(cast(Any, value)) for key, value in cast(Mapping, state["weightFactors"]).items()
        }

    active_estimate = {"name": "joint"}

    def report_iteration(item):
        print(
            "[LlrProcessing:estimateVarianceFactors] "
            f"estimate={active_estimate['name']} "
            f"adjustmentIteration={item['adjustmentIteration']} "
            f"sigmaWeightIteration={item['sigmaWeightIteration']} "
            f"active={item['activeObservationCount']} "
            f"rejected={item['rejectedObservationCount']}",
            flush=True,
        )

    processing_results: list[dict[str, object]] = []
    equation_source = problem.equation_source
    result: LlrAdjustmentResult | None = None
    available = tuple(block.block_id for block in parametrization.blocks)
    selected = available
    estimate_steps = [step for step in plan.processing_steps if isinstance(step, EstimateStep)]
    estimate_index = 0

    for step in plan.processing_steps:
        if isinstance(step, ScreenObservationsStep):
            observation_domain = screen_observations(
                equation_source(0),
                parametrization,
                model_state=processor.model_state,
                residual=step.residual,
                reported_sigma=step.accuracy,
                variance_components=plan.variance_components,
            )
            processing_results.append(
                {
                    "type": "screenObservations",
                    "residual": {
                        "maximumAbsoluteM": step.residual.maximumAbsoluteM,
                        "maximumAbsoluteByStationM": step.residual.maximumAbsoluteByStationM,
                        "rejectedCount": len(observation_domain.gross_rejected),
                    },
                    "reportedSigma": {
                        "minimumOneWayM": step.accuracy.minimumOneWayM,
                        "minimumFractionOfGroupMedian": (step.accuracy.minimumFractionOfGroupMedian),
                        "rejectedCount": sum(
                            record["status"] == "REJECTED" for record in observation_domain.accuracy_records.values()
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
            unknown = set(step.adjustment.convergenceThresholdByParametrizations or {}) - set(selected)
            if unknown:
                raise ValueError(
                    f"Estimate step {step.name!r} has thresholds for inactive parametrizations: {sorted(unknown)}."
                )
            estimate_index += 1
            active_estimate["name"] = step.name
            estimate_parametrization = parametrization.select_blocks(selected)
            estimate_settings = step.apply(plan)
            estimate_result = LlrAdjustmentSolver(
                equation_source=equation_source,
                parametrization=estimate_parametrization,
                settings=estimate_settings,
                model_state=processor.model_state,
                utc_offset_hours=float(config["utcOffsetHours"]),
                initial_sigma_factors=previous_sigma_factors or None,
                initial_weight_factors=previous_weight_factors or None,
                observation_domain=cast(LlrAdjustmentObservationDomain, observation_domain),
                iteration_callback=report_iteration if bool(config["showProgress"]) else None,
            ).run(finalize=estimate_index == len(estimate_steps))
            previous_sigma_factors = dict(estimate_result.sigma_factors)
            previous_weight_factors = {
                int(cast(Any, key)): float(value) for key, value in estimate_result.weight_factors.items()
            }
            observation_domain = estimate_result.observation_domain
            processing_results.append(
                {
                    "type": "estimate",
                    "name": step.name,
                    "parametrizations": [block.block_id for block in estimate_parametrization.blocks],
                    "settings": {
                        "observationScreening": {
                            "residual": asdict(estimate_settings.residual),
                            "reportedSigma": asdict(estimate_settings.accuracy_screening),
                        },
                        "estimate": asdict(estimate_settings.adjustment),
                        "robustWeighting": asdict(estimate_settings.robust_weights),
                    },
                    "summary": estimate_result.summary,
                    "state": estimate_result.state,
                }
            )
            if estimate_index == len(estimate_steps):
                result = cast(LlrAdjustmentResult, estimate_result)
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
            from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
            from lunarops.fileio.covariance import write_covariance
            from lunarops.fileio.parameter_vectors import write_parameter_vector

            processing_results.append(
                {
                    "type": "writeResults",
                    "outputFileReport": step.output_file_report,
                    "outputFileState": step.output_file_state,
                    "outputFileSolution": step.output_file_solution,
                    "outputFileCovariance": step.output_file_covariance,
                    "outputFileReflectorCatalog": step.output_file_reflector_catalog,
                    "outputFileStationCatalog": step.output_file_station_catalog,
                }
            )
            if step.output_file_report:
                report_payload = result.to_dict()
                report_payload.update(
                    {
                        "fingerprint": fingerprint,
                        "scientificInputs": {
                            **problem.manifest,
                            "observationGmM3S2": dict(GM_BY_BODY),
                        },
                        "execution": {
                            "mode": "mpi" if context.runtime is not None else "serial",
                            "workerCount": context.runtime.size - 1 if context.runtime is not None else 0,
                        },
                        "processingSteps": processing_results,
                        "finalRemainingCorrection": {
                            str(name): float(value)
                            for name, value in zip(result.normals.parameter_names, result.remaining_correction)
                        },
                    }
                )
                write_processing_report(context.resolve_path(step.output_file_report), report_payload)
            if step.output_file_state:
                state_payload = {
                    "fingerprint": fingerprint,
                    "lastEstimate": active_estimate["name"],
                    "converged": result.converged,
                    "parametrization": parametrization.state(),
                    "reflectorPositions": processor.model_state.reflector_positions_pa_m(),
                    "stationStates": processor.model_state.station_states_itrf_m(),
                    "sigmaFactors": result.sigma_factors,
                    "weightFactors": {str(key): float(value) for key, value in result.weight_factors.items()},
                }
                write_processing_state(context.resolve_path(step.output_file_state), state_payload)
            if (
                step.output_file_solution
                or step.output_file_covariance
                or step.output_file_reflector_catalog
                or step.output_file_station_catalog
            ):
                result.require_formal_product()
            if step.output_file_solution:
                write_parameter_vector(result.parameter_vector(), context.resolve_path(step.output_file_solution))
            if step.output_file_covariance:
                write_covariance(result.covariance_matrix(), context.resolve_path(step.output_file_covariance))
            if step.output_file_reflector_catalog:
                write_reflector_catalog(
                    processor.model_state.reflector_catalog,
                    context.resolve_path(step.output_file_reflector_catalog),
                )
            if step.output_file_station_catalog:
                write_station_catalog(
                    processor.model_state.station_catalog,
                    context.resolve_path(step.output_file_station_catalog),
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
