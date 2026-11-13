"""Worker side of the task runtimes: the models built from broadcast specs and the task handlers.

Every worker process, an MPI rank (``MpiRuntime``) or a local process (``ProcessRuntime``), keeps one
``WorkerState``. A broadcast spec (an observation model, a predictor or lunar-dynamics epoch inputs) builds its
model once with ``WorkerState.initialize``; each task then names its spec by ID and is evaluated by the handler
of its kind in ``TASK_HANDLERS``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from lunarops.config.context import ClassCache, RunContext
from lunarops.parallel.observation_spec import (
    apply_catalog_state,
    build_worker_observation_runtime,
    build_worker_processor,
)

FLUSH_TASK = "flush"
"""Task kind that settles the workers after a stopped stream (see ``MpiRuntime._settle``)."""

SPEC_LABELS = {
    "observationSpec": "observation model",
    "prediction": "prediction model",
    "epochInputs": "epoch-input prefetch",
}
"""Names of the spec kinds in run output."""


@dataclass
class WorkerState:
    """Models built by one worker from the broadcast specs, keyed by spec ID.

    Attributes
    ----------
    specs : dict of int to dict
        Received specs.
    contexts : dict of int to RunContext
        Worker run context of each spec.
    processors : dict of int to LlrObservationProcessor
        Observation processors of ``observationSpec`` specs.
    predictors : dict of int to LlrObservationPredictor
        Predictors of ``prediction`` specs.
    ephemerides : dict of int to object
        Ephemerides of ``epochInputs`` specs.
    class_cache : ClassCache
        Model instances shared by all specs of the worker.
    """

    specs: dict[int, dict] = field(default_factory=dict)
    contexts: dict[int, RunContext] = field(default_factory=dict)
    processors: dict[int, Any] = field(default_factory=dict)
    predictors: dict[int, Any] = field(default_factory=dict)
    ephemerides: dict[int, Any] = field(default_factory=dict)
    class_cache: ClassCache = field(default_factory=ClassCache)

    def initialize(self, kind: str, spec_id: int, spec: dict) -> None:
        """Build the model of a received spec.

        Parameters
        ----------
        kind : str
            ``observationSpec`` (observation processor), ``prediction`` (predictor) or any other kind,
            which opens the ephemeris of an epoch-inputs spec.
        spec_id : int
            Spec ID assigned by the master.
        spec : dict
            Spec contents.
        """
        self.specs[spec_id] = spec
        if kind == "observationSpec":
            self.contexts[spec_id], self.processors[spec_id] = build_worker_processor(spec, self.class_cache)
        elif kind == "prediction":
            from lunarops.classes.observation import LlrObservationPredictor, PredictionCriteria, PredictionMeteorology

            context, runtime = build_worker_observation_runtime(spec, self.class_cache)
            self.contexts[spec_id] = context
            options = spec["prediction"]
            station, reflector = options["station"], options["reflector"]
            self.predictors[spec_id] = LlrObservationPredictor(
                runtime.frames,
                runtime.light_time_solver,
                runtime.assembly.station_catalog[station],
                runtime.assembly.reflector_catalog[reflector],
                station_key=station,
                reflector_key=reflector,
                criteria=PredictionCriteria(**options["criteria"]),
                meteorology=PredictionMeteorology(**options["meteorology"]),
                utc_offset_hours=options["utcOffsetHours"],
            )
        else:
            context = RunContext(working_dir=spec["workingDir"], class_cache=self.class_cache)
            self.contexts[spec_id] = context
            self.ephemerides[spec_id] = context.create_class("ephemerides", spec["ephemerides"])

    def close(self) -> None:
        """Close the worker contexts and the shared class cache."""
        with ExitStack() as resources:
            resources.callback(self.class_cache.close)
            for context in self.contexts.values():
                resources.callback(context.close)


def _handle_observation_equations(payload: dict, state: WorkerState):
    """Evaluate a chunk of normal points into observation equations or result-table rows."""
    from lunarops.classes.observation import (
        ObservationOptions,
        ObservationResultDetail,
    )
    from lunarops.classes.observation.normal_points import NptDataset

    spec = state.specs[payload["specId"]]
    processor = state.processors[payload["specId"]]
    apply_catalog_state(processor, payload.get("catalogState"))
    options = ObservationOptions(**payload["options"]).with_progress(
        None,
        enabled=False,
    )
    start = payload["startIndex"]
    records = spec["datasets"][payload["sourceName"]].records[start : start + payload["recordCount"]]
    local = NptDataset(
        records=records,
        name=f"mpi-task-{payload['taskId']}",
        n_input_records=len(records),
        n_invalid_records=0,
    )
    response = {
        "sourceName": payload["sourceName"],
        "startIndex": payload["startIndex"],
        "nRecords": len(records),
    }
    if payload["returnRows"]:
        level = ObservationResultDetail.parse(payload["outputLevel"])
        response["items"] = processor.rows(local, options=options, detail=level)
    else:
        response["items"] = processor.equations(local, options=options)
    return response


def _handle_prediction(payload: dict, state: WorkerState):
    """Evaluate the prediction rows of grid indices ``startIndex`` to ``startIndex + recordCount - 1``."""
    predictor = state.predictors[payload["specId"]]
    items = []
    for index in range(payload["startIndex"], payload["startIndex"] + payload["recordCount"]):
        epoch = payload["startEpoch"].shifted(index * payload["stepSeconds"])
        items.append({"index": index, "row": predictor.evaluate(epoch)})
    return {"nRecords": len(items), "items": items}


def _handle_epoch_data(payload: dict, state: WorkerState):
    """Compute the epoch data of a batch of offsets from the initial epoch."""
    from lunarops.classes.dynamics.epoch_data import batch_epochs

    spec_id = payload["specId"]
    epochs = batch_epochs(payload["initialEpoch"], payload["offsets"])
    return state.specs[spec_id]["inputs"].build(state.ephemerides[spec_id], epochs)


#: Worker task handlers keyed by task kind.
TASK_HANDLERS: dict[str, Callable[[dict, WorkerState], object]] = {
    "observation_equations": _handle_observation_equations,
    "prediction": _handle_prediction,
    "epoch_data": _handle_epoch_data,
    # Answered in order behind any queued task, so that the master can discard a stopped stream (see
    # MpiRuntime._settle).
    FLUSH_TASK: lambda payload, state: None,
}
