"""Nonlinear LLR processing with an explicit, ordered list of processing steps.

The ``LlrProcessing`` program screens the normal points, estimates the selected parameter blocks by iterated
linearized least squares with variance-component and robust-weight estimation, and writes the results named
by its output steps. A processing state written by one run can be continued by a later run with the same
fingerprint.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from lunarops import console
from lunarops.classes.observation.assembly import build_observation_processor
from lunarops.config.context import RunContext
from lunarops.config.fingerprints import model_compatibility_fingerprint, processing_fingerprint
from lunarops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    load_datasets,
)
from lunarops.programs.registry import ArtifactSlot, program, program_spec
from lunarops.programs.specs import observation_fields, validate_processing_config

if TYPE_CHECKING:
    from lunarops.classes.observation import LlrObservationProcessor
    from lunarops.classes.parametrization.base import ParametrizationList
    from lunarops.estimation.adjustment_plan import LlrAdjustmentPlan
    from lunarops.llr_workflow import EquationSource


@dataclass(frozen=True, slots=True)
class ProcessingProblem:
    """Prepared inputs of a processing run.

    Attributes
    ----------
    plan : LlrAdjustmentPlan
        Variance components and processing steps parsed from the config.
    datasets : dict of str to NptDataset
        Normal-point datasets keyed by file stem.
    parametrization : ParametrizationList
        All parameter blocks of the config.
    processor : LlrObservationProcessor
        Observation processor holding the a-priori model state.
    equation_source : EquationSource
        Callable that evaluates the observation equations of all datasets.
    fingerprint : str
        Processing fingerprint (hexadecimal SHA-256).
    manifest : dict
        Manifest of the scientific inputs.
    """

    plan: LlrAdjustmentPlan
    datasets: dict
    parametrization: ParametrizationList
    processor: LlrObservationProcessor
    equation_source: EquationSource
    fingerprint: str
    manifest: dict[str, object]


def build_processing_problem(config, context) -> ProcessingProblem:
    """Parse the plan, read the normal points and build the models of a processing config.

    The input manifest and the fingerprint leave out ``inputFileProcessingState``, so a continued run has the
    fingerprint of the run it continues.

    Parameters
    ----------
    config : dict
        Resolved ``LlrProcessing`` config.
    context : RunContext
        Run context.

    Returns
    -------
    ProcessingProblem
        The prepared inputs.
    """
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
    """Apply the reflector positions (PA, m), station states (ITRF, m) and block values of a saved state."""
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
    """Write the final per-observation results, grouped by source file, in snake_case columns."""
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
    columns = {name: re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower() for name, _, _ in OBSERVATION_REPORT_FIELDS}
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
    program_spec(
        name="LlrProcessing",
        summary="Run nonlinear LLR estimation and explicit processing output steps.",
        inputs=(
            ArtifactSlot(
                "inputFilesNormalPoints",
                "normalPoint",
                many=True,
                description="Canonical normal-point files, or directories searched recursively for them.",
            ),
            ArtifactSlot(
                "inputFileProcessingState",
                "processingState",
                required=False,
                description=(
                    "Processing state of an earlier run to continue from. Its scientific fingerprint must match the "
                    "current inputs and models."
                ),
            ),
            ArtifactSlot(
                "inputFileStationCatalog",
                "stationCatalog",
                description="Native ITRF station catalog (a-priori values).",
            ),
            ArtifactSlot(
                "inputFileReflectorCatalog",
                "reflectorCatalog",
                description="Native Moon-PA reflector catalog (a-priori values).",
            ),
            ArtifactSlot(
                "inputFileEarthOrientationUncertainty",
                "ExternalEarthOrientationParameterFile",
                external=True,
                required=False,
                description=(
                    "IERS 20 C04 daily EOP file whose UT1 and polar-motion uncertainties are the prior sigmas of "
                    "the Earth-orientation corrections (`earthOrientationCorrection`)."
                ),
            ),
        ),
        fields=observation_fields(parametrized=True, processing=True),
        validator=validate_processing_config,
    )
)
def llr_processing(config: dict, context: RunContext):
    """Estimate parameters from LLR normal points by running the steps of ``processingSteps`` in order.

    Reads the ``normalPoint`` files ``inputFilesNormalPoints`` and the a-priori ``stationCatalog`` and
    ``reflectorCatalog``. With ``inputFileProcessingState``, the parameter values, reflector positions,
    station states, sigma factors and weight factors of an earlier run are restored; its fingerprint must
    equal the current one. The steps then run in order:

    - ``screenObservations`` rejects normal points by prefit residual and reported sigma;
    - ``selectParametrizations`` sets the parameter blocks of the following estimates;
    - ``estimate`` runs the nonlinear adjustment, starting from the sigma and weight factors of the
      previous estimate; the last estimate gives the final result;
    - ``writeResiduals`` writes the final per-observation results (``observationResult``);
    - ``writeNormalEquations`` writes the final normal equations;
    - ``writeResults`` writes any of the processing report, the processing state, the parameter vector,
      the covariance matrix and the updated reflector and station catalogs.

    See the program reference (``docs/reference/programs.md``) for every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    LlrAdjustmentResult
        Result of the final estimate.

    Raises
    ------
    ValueError
        If the processing-state fingerprint does not match, or an estimate step has no parameter blocks or
        thresholds for blocks it does not estimate.
    """
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
    previous_floor_sigmas: dict[str, float] = {}
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
        previous_floor_sigmas = {
            str(key): float(cast(Any, value)) for key, value in cast(Mapping, state.get("floorSigmasM") or {}).items()
        }
        previous_weight_factors = {
            int(cast(Any, key)): float(cast(Any, value)) for key, value in cast(Mapping, state["weightFactors"]).items()
        }

    active_estimate = {"name": "joint"}
    iteration_table = {"table": console.Table(())}
    written: list[str] = []

    def output_path(path):
        resolved = context.resolve_path(path)
        written.append(str(resolved))
        return resolved

    def report_iteration(item):
        wrms = item.get("candidateWrmsM")
        iteration_table["table"].row(
            item["adjustmentIteration"],
            item["sigmaWeightIteration"],
            item["activeObservationCount"],
            item["rejectedObservationCount"],
            None if wrms is None else 1e3 * float(cast(Any, wrms)),
            item.get("maximumCandidateParameterUpdateM"),
            item.get("maximumSigmaFactorChange"),
        )

    processing_results: list[dict[str, object]] = []
    equation_source = problem.equation_source
    result: LlrAdjustmentResult | None = None
    available = tuple(block.block_id for block in parametrization.blocks)
    selected = available
    estimate_steps = [step for step in plan.processing_steps if isinstance(step, EstimateStep)]
    estimate_index = 0
    earth_orientation_uncertainty = None
    if any(step.adjustment.earthOrientationCorrection for step in estimate_steps):
        from lunarops.estimation.earth_orientation_effects import EarthOrientationUncertainty

        if not config.get("inputFileEarthOrientationUncertainty"):
            raise ValueError("earthOrientationCorrection needs inputFileEarthOrientationUncertainty.")
        earth_orientation_uncertainty = EarthOrientationUncertainty.from_c04(
            context.resolve_path(config["inputFileEarthOrientationUncertainty"])
        )

    for step in plan.processing_steps:
        if isinstance(step, ScreenObservationsStep):
            observation_domain = screen_observations(
                equation_source(0),
                parametrization,
                residual=step.residual,
                reported_sigma=step.accuracy,
                variance_components=plan.variance_components,
            )
            residual_rejected = len(observation_domain.gross_rejected)
            sigma_rejected = sum(
                record["status"] == "REJECTED" for record in observation_domain.accuracy_records.values()
            )
            retained = len(observation_domain.retained_keys)
            processing_results.append(
                {
                    "type": "screenObservations",
                    "residual": {
                        "maximumAbsoluteM": step.residual.maximumAbsoluteM,
                        "maximumAbsoluteByStationM": step.residual.maximumAbsoluteByStationM,
                        "rejectedCount": residual_rejected,
                    },
                    "reportedSigma": {
                        "minimumOneWayM": step.accuracy.minimumOneWayM,
                        "minimumFractionOfGroupMedian": (step.accuracy.minimumFractionOfGroupMedian),
                        "rejectedCount": sigma_rejected,
                    },
                    "retainedCount": retained,
                }
            )
            console.info(
                f"screening: {retained:,} normal points kept, {residual_rejected + sigma_rejected:,} rejected "
                f"({residual_rejected:,} by residual, {sigma_rejected:,} by reported sigma)"
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
            console.info(
                f"estimate {step.name!r} ({estimate_index}/{len(estimate_steps)}), parameters: "
                f"{', '.join(block.block_id for block in estimate_parametrization.blocks)}"
            )
            iteration_table["table"] = console.Table(
                (
                    console.Column("iter", 4),
                    console.Column("reweight", 8),
                    console.Column("used", 7),
                    console.Column("rejected", 8),
                    console.Column("WRMS [mm]", 9, format=".3f"),
                    console.Column("max update [m]", 14, format=".3e"),
                    console.Column("max sigma change", 16, format=".2e"),
                )
            )
            estimate_result = LlrAdjustmentSolver(
                equation_source=equation_source,
                parametrization=estimate_parametrization,
                settings=estimate_settings,
                utc_offset_hours=float(config["utcOffsetHours"]),
                initial_sigma_factors=previous_sigma_factors or None,
                initial_floor_sigmas=previous_floor_sigmas or None,
                initial_weight_factors=previous_weight_factors or None,
                observation_domain=cast(LlrAdjustmentObservationDomain, observation_domain),
                iteration_callback=report_iteration if bool(config["showProgress"]) else None,
                earth_orientation_uncertainty=earth_orientation_uncertainty,
                station_positions={
                    key: record.itrf_xyz_m for key, record in processor.model_state.station_catalog.items()
                },
            ).run(finalize=estimate_index == len(estimate_steps))
            previous_sigma_factors = dict(estimate_result.sigma_factors)
            previous_floor_sigmas = dict(estimate_result.floor_sigmas)
            previous_weight_factors = {
                int(cast(Any, key)): float(value) for key, value in estimate_result.weight_factors.items()
            }
            observation_domain = estimate_result.observation_domain
            iteration_count = len(estimate_result.adjustment_iterations)
            iterations = f"{iteration_count} iteration" + ("" if iteration_count == 1 else "s")
            if estimate_result.converged:
                console.info(f"estimate {step.name!r}: converged after {iterations}")
            else:
                console.warning(f"estimate {step.name!r}: not converged, iteration limit reached ({iterations})")
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
            written.append(str(context.resolve_path(step.output_file)))
            processing_results.append({"type": "writeResiduals", "outputFile": step.output_file})
            continue
        if isinstance(step, WriteNormalEquationsStep):
            from lunarops.fileio.normal_equations import write_normal_equations

            write_normal_equations(result.normals, output_path(step.output_file))
            processing_results.append({"type": "writeNormalEquations", "outputFile": step.output_file})
            continue
        if isinstance(step, WriteResultsStep):
            from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
            from lunarops.fileio.covariance import write_covariance
            from lunarops.fileio.parameter_vectors import write_parameter_vector

            processing_results.append({"type": "writeResults", **asdict(step)})
            if step.outputFileReport:
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
                write_processing_report(output_path(step.outputFileReport), report_payload)
            if step.outputFileState:
                state_payload = {
                    "fingerprint": fingerprint,
                    "lastEstimate": active_estimate["name"],
                    "converged": result.converged,
                    "parametrization": parametrization.state(),
                    "reflectorPositions": processor.model_state.reflector_positions_pa_m(),
                    "stationStates": processor.model_state.station_states_itrf_m(),
                    "sigmaFactors": result.sigma_factors,
                    "floorSigmasM": result.floor_sigmas,
                    "weightFactors": {str(key): float(value) for key, value in result.weight_factors.items()},
                }
                write_processing_state(output_path(step.outputFileState), state_payload)
            if (
                step.outputFileSolution
                or step.outputFileCovariance
                or step.outputFileReflectorCatalog
                or step.outputFileStationCatalog
            ):
                result.require_formal_product()
            if step.outputFileSolution:
                write_parameter_vector(result.parameter_vector(), output_path(step.outputFileSolution))
            if step.outputFileCovariance:
                write_covariance(result.covariance_matrix(), output_path(step.outputFileCovariance))
            if step.outputFileReflectorCatalog:
                write_reflector_catalog(
                    processor.model_state.reflector_catalog,
                    output_path(step.outputFileReflectorCatalog),
                )
            if step.outputFileStationCatalog:
                write_station_catalog(
                    processor.model_state.station_catalog,
                    output_path(step.outputFileStationCatalog),
                )
            continue
        raise TypeError(f"Unsupported processing step {type(step).__name__}.")

    if result is None:
        raise RuntimeError("LlrProcessing produced no final estimate.")
    sigma0 = result.summary.get("sigma0Post")
    console.summary(
        {
            **({"final estimate": active_estimate["name"]} if len(estimate_steps) > 1 else {}),
            "converged": result.converged,
            "iterations": len(result.adjustment_iterations),
            "reweighting iterations": len(result.sigma_weight_iterations),
            **({"a posteriori sigma0": float(cast(Any, sigma0))} if isinstance(sigma0, (int, float)) else {}),
        },
        outputs=written,
    )
    return result


__all__ = ["llr_processing"]
