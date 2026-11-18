"""Nonlinear LLR processing with an explicit, ordered list of processing steps.

The ``LlrProcessing`` program screens the normal points, estimates the selected parameter blocks by iterated
linearized least squares with variance-component and robust-weight estimation, and writes the results named
by its output steps. A processing state written by one run can be continued by a later run with the same
fingerprint.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from lunarops import console
from lunarops.classes.dynamics.parameters import write_lunar_dynamics_parameters
from lunarops.classes.frames.coordinate_scaling import GM_BY_BODY
from lunarops.classes.observation import LlrObservationProcessor
from lunarops.classes.observation.assembly import build_observation_processor
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.classes.parametrization.lunar_dynamics import LunarDynamicsParametrization
from lunarops.config.context import RunContext
from lunarops.estimation.adjustment.plan import (
    EstimateStep,
    LlrAdjustmentPlan,
    ScreenObservationsStep,
    SelectParametrizationsStep,
    WriteNormalEquationsStep,
    WriteResidualsStep,
    WriteResultsStep,
    parse_adjustment_plan,
)
from lunarops.estimation.adjustment.reporting import OBSERVATION_REPORT_FIELDS
from lunarops.estimation.adjustment.results import LlrAdjustmentObservationDomain, LlrAdjustmentResult
from lunarops.estimation.adjustment.screening import screen_observations
from lunarops.estimation.adjustment.solver import LlrAdjustmentSolver
from lunarops.estimation.earth_orientation import EarthOrientationUncertainty
from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
from lunarops.fileio.covariance import write_covariance
from lunarops.fileio.normal_equations import write_normal_equations
from lunarops.fileio.observation_results import write_observation_results
from lunarops.fileio.parameter_vectors import write_parameter_vector
from lunarops.fileio.processing_artifacts import read_processing_state, write_processing_state
from lunarops.fileio.structured_reports import write_processing_report
from lunarops.llr_workflow import (
    EquationSource,
    build_equation_source,
    build_parametrization,
    load_datasets,
)
from lunarops.programs.observation_schema import TASKS_REMOVED_KEYS, observation_fields, validate_processing_config
from lunarops.programs.provenance import (
    model_compatibility_fingerprint,
    processing_fingerprint,
    scientific_input_manifest,
)
from lunarops.programs.registry import ArtifactSlot, program, program_spec

#: Name of the program; it selects the spec that classifies the config keys of the fingerprints.
_PROGRAM_NAME = "LlrProcessing"

#: Output files of the output steps, by step type; each key is a field of the step's config entry.
_STEP_OUTPUT_SLOTS: dict[str, tuple[ArtifactSlot, ...]] = {
    "writeResiduals": (ArtifactSlot("outputFile", "observationResult"),),
    "writeNormalEquations": (ArtifactSlot("outputFile", "normalEquations", directory=True),),
    "writeResults": (
        ArtifactSlot("outputFileReport", "processingReport"),
        ArtifactSlot("outputFileState", "processingState"),
        ArtifactSlot("outputFileSolution", "parameterVector"),
        ArtifactSlot("outputFileCovariance", "covariance", directory=True),
        ArtifactSlot("outputFileReflectorCatalog", "reflectorCatalog"),
        ArtifactSlot("outputFileStationCatalog", "stationCatalog"),
        ArtifactSlot("outputFileLunarDynamicsParameters", "lunarDynamicsParameters"),
    ),
}


def _processing_step_outputs(config: Mapping[str, object]) -> Iterator[tuple[ArtifactSlot, object]]:
    """Yield a slot named ``processingSteps[i].<key>`` and the path of each file written by the output steps."""
    for index, step in enumerate(cast(list[dict], config["processingSteps"])):
        for slot in _STEP_OUTPUT_SLOTS.get(step["type"], ()):
            if step.get(slot.key) is not None:
                yield replace(slot, key=f"processingSteps[{index}].{slot.key}"), step[slot.key]


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
    manifest = scientific_input_manifest(
        {key: value for key, value in config.items() if key != "inputFileProcessingState"},
        context,
        _PROGRAM_NAME,
    )
    plan = parse_adjustment_plan(config)
    datasets = load_datasets(config, context)
    processor = build_observation_processor(context, config)
    parametrization = build_parametrization(config, context, processor.model_state, processor.observation_model)
    return ProcessingProblem(
        plan,
        datasets,
        parametrization,
        processor,
        build_equation_source(config, context, datasets, processor),
        processing_fingerprint(config, context, program=_PROGRAM_NAME, plan=plan, manifest=manifest),
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


def _float_mapping(values: object) -> dict[str, float]:
    """Return a saved ``{name: number}`` mapping with string keys and float values."""
    return {str(key): float(value) for key, value in cast(Mapping[Any, Any], values).items()}


def _write_residuals(step, result, datasets, context: RunContext) -> None:
    """Write the final per-observation results, grouped by source file, in snake_case columns."""
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


def _iteration_table() -> console.Table:
    """Return the console table of the adjustment iterations of one estimate."""
    return console.Table(
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


class _ProcessingRun:
    """State carried from one processing step to the next.

    An estimate starts from the sigma, floor, session and weight factors and the observation domain left by
    the previous estimate (or by a restored processing state); the output steps write the final estimate.
    """

    def __init__(self, config: dict, context: RunContext, problem: ProcessingProblem) -> None:
        self.config = config
        self.context = context
        self.problem = problem
        self.available = tuple(block.block_id for block in problem.parametrization.blocks)
        self.selected = self.available
        self.estimate_count = sum(isinstance(step, EstimateStep) for step in problem.plan.processing_steps)
        self.estimate_index = 0
        self.estimate_name = "joint"
        self.iteration_table = console.Table(())
        self.sigma_factors: dict[str, float] = {}
        self.floor_sigmas: dict[str, float] = {}
        self.session_sigmas: dict[str, float] = {}
        self.weight_factors: dict[Hashable, float] = {}
        self.observation_domain: LlrAdjustmentObservationDomain | None = None
        self.result: LlrAdjustmentResult | None = None
        self.processing_results: list[dict[str, object]] = []
        self.written: list[str] = []
        self.earth_orientation_uncertainty = self._earth_orientation_uncertainty()

    @classmethod
    def from_state(
        cls, state: Mapping[str, object], config: dict, context: RunContext, problem: ProcessingProblem
    ) -> _ProcessingRun:
        """Continue a processing run from a saved processing state with the same fingerprint."""
        if state["fingerprint"] != problem.fingerprint:
            raise ValueError("Processing-state fingerprint does not match the current inputs and model configuration.")
        _restore_state(state, problem.parametrization, problem.processor)
        run = cls(config, context, problem)
        run.sigma_factors = _float_mapping(state["sigmaFactors"])
        run.floor_sigmas = _float_mapping(state.get("floorSigmasM") or {})
        run.session_sigmas = _float_mapping(state.get("sessionSigmasM") or {})
        run.weight_factors = {
            int(key): float(value) for key, value in cast(Mapping[Any, Any], state["weightFactors"]).items()
        }
        return run

    def _earth_orientation_uncertainty(self) -> EarthOrientationUncertainty | None:
        """Read the EOP uncertainties when an estimate step corrects Earth orientation."""
        if not any(
            step.adjustment.earthOrientationCorrection
            for step in self.problem.plan.processing_steps
            if isinstance(step, EstimateStep)
        ):
            return None
        if not self.config.get("inputFileEarthOrientationUncertainty"):
            raise ValueError("earthOrientationCorrection needs inputFileEarthOrientationUncertainty.")
        return EarthOrientationUncertainty.from_c04(
            self.context.resolve_path(self.config["inputFileEarthOrientationUncertainty"])
        )

    def run(self, step: object) -> None:
        """Run one processing step."""
        handler = _STEP_HANDLERS.get(type(step))
        if handler is None:
            raise TypeError(f"Unsupported processing step {type(step).__name__}.")
        handler(self, step)

    def _output_path(self, path: object) -> Path:
        resolved = self.context.resolve_path(path)
        self.written.append(str(resolved))
        return resolved

    def _final_result(self) -> LlrAdjustmentResult:
        if self.result is None:
            raise RuntimeError("Output steps require a completed final estimate.")
        return self.result

    def _report_iteration(self, item: Mapping[str, Any]) -> None:
        wrms = item.get("candidateWrmsM")
        self.iteration_table.row(
            item["adjustmentIteration"],
            item["sigmaWeightIteration"],
            item["activeObservationCount"],
            item["rejectedObservationCount"],
            None if wrms is None else 1e3 * float(wrms),
            item.get("maximumCandidateParameterUpdateM"),
            item.get("maximumSigmaFactorChange"),
        )

    def screen_observations(self, step: ScreenObservationsStep) -> None:
        """Reject normal points by prefit residual and reported sigma."""
        domain = screen_observations(
            self.problem.equation_source(0),
            self.problem.parametrization,
            residual=step.residual,
            reported_sigma=step.accuracy,
            variance_components=self.problem.plan.variance_components,
        )
        self.observation_domain = domain
        residual_rejected = len(domain.gross_rejected)
        sigma_rejected = sum(record["status"] == "REJECTED" for record in domain.accuracy_records.values())
        retained = len(domain.retained_keys)
        self.processing_results.append(
            {
                "type": "screenObservations",
                "residual": {
                    "maximumAbsoluteM": step.residual.maximumAbsoluteM,
                    "maximumAbsoluteByStationM": step.residual.maximumAbsoluteByStationM,
                    "rejectedCount": residual_rejected,
                },
                "reportedSigma": {
                    "minimumOneWayM": step.accuracy.minimumOneWayM,
                    "minimumFractionOfGroupMedian": step.accuracy.minimumFractionOfGroupMedian,
                    "rejectedCount": sigma_rejected,
                },
                "retainedCount": retained,
            }
        )
        console.info(
            f"screening: {retained:,} normal points kept, {residual_rejected + sigma_rejected:,} rejected "
            f"({residual_rejected:,} by residual, {sigma_rejected:,} by reported sigma)"
        )

    def select_parametrizations(self, step: SelectParametrizationsStep) -> None:
        """Set the parameter blocks of the following estimates."""
        self.selected = step.apply(self.available)
        self.processing_results.append({"type": "selectParametrizations", "parametrizations": list(self.selected)})

    def estimate(self, step: EstimateStep) -> None:
        """Run the nonlinear adjustment of the selected blocks; the last estimate gives the final result."""
        if not self.selected:
            raise ValueError(f"Estimate step {step.name!r} has no enabled parametrizations.")
        unknown = set(step.adjustment.convergenceThresholdByParametrizations or {}) - set(self.selected)
        if unknown:
            raise ValueError(
                f"Estimate step {step.name!r} has thresholds for inactive parametrizations: {sorted(unknown)}."
            )
        self.estimate_index += 1
        self.estimate_name = step.name
        is_final = self.estimate_index == self.estimate_count
        parametrization = self.problem.parametrization.select_blocks(self.selected)
        settings = step.apply(self.problem.plan)
        console.info(
            f"estimate {step.name!r} ({self.estimate_index}/{self.estimate_count}), parameters: "
            f"{', '.join(block.block_id for block in parametrization.blocks)}"
        )
        self.iteration_table = _iteration_table()
        estimate_result = LlrAdjustmentSolver(
            equation_source=self.problem.equation_source,
            parametrization=parametrization,
            settings=settings,
            utc_offset_hours=float(self.config["utcOffsetHours"]),
            initial_sigma_factors=self.sigma_factors or None,
            initial_floor_sigmas=self.floor_sigmas or None,
            initial_session_sigmas=self.session_sigmas or None,
            initial_weight_factors=self.weight_factors or None,
            observation_domain=cast(LlrAdjustmentObservationDomain, self.observation_domain),
            iteration_callback=self._report_iteration if bool(self.config["showProgress"]) else None,
            earth_orientation_uncertainty=self.earth_orientation_uncertainty,
            station_positions={
                key: record.itrf_xyz_m for key, record in self.problem.processor.model_state.station_catalog.items()
            },
        ).run(finalize=is_final)
        self.sigma_factors = dict(estimate_result.sigma_factors)
        self.floor_sigmas = dict(estimate_result.floor_sigmas)
        # A step without night sessions keeps the session sigmas of an earlier one for a later step.
        self.session_sigmas.update(estimate_result.session_sigmas)
        self.weight_factors = {
            int(cast(Any, key)): float(value) for key, value in estimate_result.weight_factors.items()
        }
        self.observation_domain = estimate_result.observation_domain
        iteration_count = len(estimate_result.adjustment_iterations)
        iterations = f"{iteration_count} iteration" + ("" if iteration_count == 1 else "s")
        if estimate_result.converged:
            console.info(f"estimate {step.name!r}: converged after {iterations}")
        else:
            console.warning(f"estimate {step.name!r}: not converged, iteration limit reached ({iterations})")
        if not estimate_result.summary["stochasticModelConverged"]:
            console.warning(
                f"estimate {step.name!r}: the solve/reweight passes of the last linearization reached their limit"
            )
        self.processing_results.append(
            {
                "type": "estimate",
                "name": step.name,
                "parametrizations": [block.block_id for block in parametrization.blocks],
                "settings": {
                    "observationScreening": {
                        "residual": asdict(settings.residual),
                        "reportedSigma": asdict(settings.accuracy_screening),
                    },
                    "estimate": asdict(settings.adjustment),
                    "robustWeighting": asdict(settings.robust_weights),
                },
                "summary": estimate_result.summary,
                "state": estimate_result.state,
            }
        )
        if is_final:
            self.result = cast(LlrAdjustmentResult, estimate_result)
            self.result.normals.meta["compatibility"] = model_compatibility_fingerprint(
                self.config, self.context, _PROGRAM_NAME
            )

    def write_residuals(self, step: WriteResidualsStep) -> None:
        """Write the final per-observation results (``observationResult``)."""
        _write_residuals(step, self._final_result(), self.problem.datasets, self.context)
        self.written.append(str(self.context.resolve_path(step.output_file)))
        self.processing_results.append({"type": "writeResiduals", "outputFile": step.output_file})

    def write_normal_equations(self, step: WriteNormalEquationsStep) -> None:
        """Write the final normal equations."""
        write_normal_equations(self._final_result().normals, self._output_path(step.output_file))
        self.processing_results.append({"type": "writeNormalEquations", "outputFile": step.output_file})

    def write_results(self, step: WriteResultsStep) -> None:
        """Write the requested report, state, solution, covariance, catalogs and dynamics parameters."""
        result = self._final_result()
        model_state = self.problem.processor.model_state
        self.processing_results.append({"type": "writeResults", **asdict(step)})
        if step.outputFileReport:
            write_processing_report(self._output_path(step.outputFileReport), self._report_payload(result))
        if step.outputFileState:
            state_payload = {
                "fingerprint": self.problem.fingerprint,
                "lastEstimate": self.estimate_name,
                "converged": result.converged,
                "parametrization": self.problem.parametrization.state(),
                "reflectorPositions": model_state.reflector_positions_pa_m(),
                "stationStates": model_state.station_states_itrf_m(),
                "sigmaFactors": result.sigma_factors,
                "floorSigmasM": result.floor_sigmas,
                "sessionSigmasM": self.session_sigmas,
                "weightFactors": {str(key): float(value) for key, value in result.weight_factors.items()},
            }
            write_processing_state(self._output_path(step.outputFileState), state_payload)
        if (
            step.outputFileSolution
            or step.outputFileCovariance
            or step.outputFileReflectorCatalog
            or step.outputFileStationCatalog
            or step.outputFileLunarDynamicsParameters
        ):
            result.require_formal_product()
        if step.outputFileSolution:
            write_parameter_vector(result.parameter_vector(), self._output_path(step.outputFileSolution))
        if step.outputFileCovariance:
            write_covariance(result.covariance_matrix(), self._output_path(step.outputFileCovariance))
        if step.outputFileReflectorCatalog:
            write_reflector_catalog(model_state.reflector_catalog, self._output_path(step.outputFileReflectorCatalog))
        if step.outputFileStationCatalog:
            write_station_catalog(model_state.station_catalog, self._output_path(step.outputFileStationCatalog))
        if step.outputFileLunarDynamicsParameters:
            dynamics_blocks = [
                block
                for block in self.problem.parametrization.blocks
                if isinstance(block, LunarDynamicsParametrization)
            ]
            if not dynamics_blocks:
                raise ValueError("outputFileLunarDynamicsParameters needs a lunarDynamics parametrization.")
            write_lunar_dynamics_parameters(
                dynamics_blocks[0].updated_parameters(),
                self._output_path(step.outputFileLunarDynamicsParameters),
            )

    def _report_payload(self, result: LlrAdjustmentResult) -> dict[str, object]:
        runtime = self.context.runtime
        payload = result.to_dict()
        payload.update(
            {
                "fingerprint": self.problem.fingerprint,
                "scientificInputs": {**self.problem.manifest, "observationGmM3S2": dict(GM_BY_BODY)},
                "execution": {
                    "mode": runtime.name if runtime is not None else "serial",
                    "workerCount": runtime.workers if runtime is not None else 0,
                },
                "processingSteps": self.processing_results,
                "finalRemainingCorrection": {
                    str(name): float(value)
                    for name, value in zip(result.normals.parameter_names, result.remaining_correction)
                },
            }
        )
        return payload


_STEP_HANDLERS: dict[type, Callable[[_ProcessingRun, Any], None]] = {
    ScreenObservationsStep: _ProcessingRun.screen_observations,
    SelectParametrizationsStep: _ProcessingRun.select_parametrizations,
    EstimateStep: _ProcessingRun.estimate,
    WriteResidualsStep: _ProcessingRun.write_residuals,
    WriteNormalEquationsStep: _ProcessingRun.write_normal_equations,
    WriteResultsStep: _ProcessingRun.write_results,
}


@program(
    program_spec(
        name=_PROGRAM_NAME,
        summary="Run nonlinear LLR estimation and explicit processing output steps.",
        nested_outputs=_processing_step_outputs,
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
        removed_keys=TASKS_REMOVED_KEYS,
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
      the covariance matrix, the updated reflector and station catalogs and the updated lunar dynamics
      parameters (nominal values plus the corrections of a ``lunarDynamics`` parametrization).

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
    problem = build_processing_problem(config, context)
    if config.get("inputFileProcessingState"):
        state = read_processing_state(context.resolve_path(config["inputFileProcessingState"]))
        run = _ProcessingRun.from_state(state, config, context, problem)
    else:
        run = _ProcessingRun(config, context, problem)
    for step in problem.plan.processing_steps:
        run.run(step)

    result = run.result
    if result is None:
        raise RuntimeError("LlrProcessing produced no final estimate.")
    sigma0 = result.summary.get("sigma0Post")
    console.summary(
        {
            **({"final estimate": run.estimate_name} if run.estimate_count > 1 else {}),
            "converged": result.converged,
            "iterations": len(result.adjustment_iterations),
            "reweighting iterations": len(result.sigma_weight_iterations),
            **({"a posteriori sigma0": float(cast(Any, sigma0))} if isinstance(sigma0, (int, float)) else {}),
        },
        outputs=run.written,
    )
    return result


__all__ = ["llr_processing"]
