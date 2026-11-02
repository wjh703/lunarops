"""Rank 0 integrates; workers evaluate tasks referencing broadcast specs."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from typing import Any

from tqdm import tqdm  # type: ignore[import-untyped]

from lunarops.config.context import ClassCache, RunContext
from lunarops.parallel.observation_spec import (
    apply_catalog_state,
    build_worker_observation_runtime,
    build_worker_processor,
)

TASK = 101
RESULT = 102
STOP = 103


def mpi_comm_world():
    try:
        from mpi4py import MPI
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "--mpi requires mpi4py and an MPI runtime; launch with "
            "mpirun/mpiexec/srun (e.g. 'srun python -m lunarops run cfg.yml --mpi')."
        ) from exc
    return MPI


@dataclass
class WorkerState:
    specs: dict[int, dict] = field(default_factory=dict)
    contexts: dict[int, RunContext] = field(default_factory=dict)
    processors: dict[int, Any] = field(default_factory=dict)
    predictors: dict[int, Any] = field(default_factory=dict)
    ephemerides: dict[int, Any] = field(default_factory=dict)
    class_cache: ClassCache = field(default_factory=ClassCache)

    def initialize(self, kind: str, spec_id: int, spec: dict) -> None:
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
                runtime.frames, runtime.light_time_solver,
                runtime.assembly.station_catalog[station], runtime.assembly.reflector_catalog[reflector],
                station_key=station, reflector_key=reflector,
                criteria=PredictionCriteria(**options["criteria"]),
                meteorology=PredictionMeteorology(**options["meteorology"]),
                utc_offset_hours=options["utcOffsetHours"],
            )
        else:
            context = RunContext(working_dir=spec["workingDir"], class_cache=self.class_cache)
            self.contexts[spec_id] = context
            self.ephemerides[spec_id] = context.create_class("ephemerides", spec["ephemerides"])

    def close(self) -> None:
        with ExitStack() as resources:
            resources.callback(self.class_cache.close)
            for context in self.contexts.values():
                resources.callback(context.close)


def _handle_observation_equations(payload: dict, state: WorkerState):
    """NptRecord chunk -> typed equations or lightweight table rows."""
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
    predictor = state.predictors[payload["specId"]]
    items = []
    for index in range(payload["startIndex"], payload["startIndex"] + payload["recordCount"]):
        epoch = payload["startEpoch"].shifted(index * payload["stepSeconds"])
        items.append({"index": index, "row": predictor.evaluate(epoch)})
    return {"nRecords": len(items), "items": items}



def _handle_epoch_data(payload: dict, state: WorkerState):
    from lunarops.classes.dynamics.epoch_data import batch_epochs

    spec_id = payload["specId"]
    epochs = batch_epochs(payload["initialEpoch"], payload["offsets"])
    return state.specs[spec_id]["inputs"].build(state.ephemerides[spec_id], epochs)


TASK_HANDLERS: dict[str, Callable[[dict, WorkerState], object]] = {
    "observation_equations": _handle_observation_equations,
    "prediction": _handle_prediction,
    "epoch_data": _handle_epoch_data,
}


class MpiRuntime:
    def __init__(self, comm=None) -> None:
        self._MPI = mpi_comm_world()
        self.comm = comm if comm is not None else self._MPI.COMM_WORLD
        self._spec_ids: dict[str, int] = {}
        if self.size < 2:
            raise RuntimeError("Explicit MPI execution requires at least two ranks (one worker).")

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        return self.comm.Get_size()

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    def broadcast_spec(self, kind: str, spec: dict) -> int:
        key = spec["specId"]
        if key in self._spec_ids:
            return self._spec_ids[key]
        spec_id = len(self._spec_ids)
        for worker in range(1, self.size):
            self.comm.send(("spec", (kind, spec_id)), dest=worker, tag=TASK)
        self.comm.bcast(spec, root=0)
        timings = self.comm.gather((0.0, ""), root=0)
        for worker, (_, error) in enumerate(timings[1:], start=1):
            if error:
                raise RuntimeError(f"MPI worker rank {worker} failed initializing {kind}:\n{error}")
        print(f"[MPI] {kind} ready on {self.size - 1} worker(s); slowest {max(t[0] for t in timings):.2f} s.",
              flush=True)
        self._spec_ids[key] = spec_id
        return spec_id

    def stream_tasks(self, kind: str, payloads: Sequence[dict], *, progress: str | None = None) -> Iterator[Any]:
        payloads = [dict(p, taskId=i) for i, p in enumerate(payloads)]
        status = self._MPI.Status()
        results: dict[int, Any] = {}
        next_task = 0
        outstanding = 0
        sends: list = []

        def send_next(worker_rank: int) -> None:
            nonlocal next_task, outstanding
            if next_task < len(payloads):
                sends.append(self.comm.isend((kind, payloads[next_task]), dest=worker_rank, tag=TASK))
                next_task += 1
                outstanding += 1
                if len(sends) > 4 * self.size:
                    sends[:] = [request for request in sends if not request.Test()]

        with tqdm(total=len(payloads), desc=progress, unit="batch", disable=progress is None) as bar:
            def receive() -> None:
                nonlocal outstanding
                is_error, task_id, result = self.comm.recv(source=self._MPI.ANY_SOURCE, tag=RESULT, status=status)
                worker = status.Get_source()
                outstanding -= 1
                if is_error:
                    raise RuntimeError(f"MPI worker rank {worker} failed on {kind} task {task_id}:\n{result}")
                results[task_id] = result
                bar.update()
                send_next(worker)

            for _ in range(2):
                for worker in range(1, self.size):
                    send_next(worker)
            try:
                for wanted in range(len(payloads)):
                    while wanted not in results:
                        receive()
                    yield results.pop(wanted)
            finally:
                next_task = len(payloads)
                while outstanding:
                    self.comm.recv(source=self._MPI.ANY_SOURCE, tag=RESULT, status=status)
                    outstanding -= 1
                self._MPI.Request.Waitall(sends)

    def map_tasks(self, kind: str, payloads: Sequence[dict], *, progress: str | None = None) -> list[Any]:
        return list(self.stream_tasks(kind, payloads, progress=progress))

    def worker_loop(self) -> None:
        status = self._MPI.Status()
        state = WorkerState()
        try:
            while True:
                message = self.comm.recv(source=0, tag=self._MPI.ANY_TAG, status=status)
                if status.Get_tag() == STOP:
                    break
                kind, payload = message
                if kind == "spec":
                    spec_kind, spec_id = payload
                    spec = self.comm.bcast(None, root=0)
                    started = time.perf_counter()
                    error = ""
                    try:
                        state.initialize(spec_kind, spec_id, spec)
                    except Exception:  # noqa: BLE001 -- forward initialization failure after collective completes
                        error = traceback.format_exc()
                    self.comm.gather((time.perf_counter() - started, error), root=0)
                else:
                    try:
                        result = TASK_HANDLERS[kind](payload, state)
                        self.comm.send((False, payload["taskId"], result), dest=0, tag=RESULT)
                    except Exception:  # noqa: BLE001 -- forward remote task failures
                        self.comm.send((True, payload["taskId"], traceback.format_exc()), dest=0, tag=RESULT)
        finally:
            state.close()

    def shutdown(self) -> None:
        for worker in range(1, self.size):
            self.comm.send(None, dest=worker, tag=STOP)
        self._spec_ids.clear()


def chunk_dataset_tasks(datasets, chunksize: int) -> list[dict]:
    chunk = int(chunksize)
    tasks: list[dict] = []
    for source_name, dataset in datasets.items():
        records = dataset.records
        for start in range(0, len(records), chunk):
            tasks.append(
                {
                    "sourceName": str(source_name),
                    "startIndex": start,
                    "recordCount": min(chunk, len(records) - start),
                }
            )
    return tasks


def _observation_task_payloads(
    spec_id: int,
    datasets,
    options,
    *,
    chunksize: int,
    catalog_state: dict | None,
    return_rows: bool = False,
    output_level: str = "standard",
) -> list[dict]:
    common = {
        "specId": spec_id,
        "options": asdict(options),
        "catalogState": catalog_state,
        "returnRows": return_rows,
        "outputLevel": output_level,
    }
    return [common | task for task in chunk_dataset_tasks(datasets, chunksize)]


def mpi_observations(
    runtime: MpiRuntime,
    spec: dict,
    datasets,
    options,
    *,
    chunksize: int = 8,
    catalog_state: dict | None = None,
    rows: bool,
    output_level: str = "standard",
    progress_desc: str = "O-C normal points",
    quiet: bool = False,
) -> dict[str, list]:
    spec.setdefault("datasets", datasets)
    spec_id = runtime.broadcast_spec("observationSpec", spec)
    tasks = _observation_task_payloads(
        spec_id,
        datasets,
        options,
        chunksize=chunksize,
        catalog_state=catalog_state,
        return_rows=rows,
        output_level=output_level,
    )
    results = runtime.map_tasks(
        "observation_equations",
        tasks,
        progress=None if quiet else progress_desc,
    )
    items_by_source: dict[str, list] = {str(name): [] for name in datasets}
    for result in results:
        source_name = result["sourceName"]
        items = result["items"]
        if rows:
            items_by_source[source_name].extend(items)
        else:
            items_by_source[source_name].append(items)
    for items in items_by_source.values():
        if rows:
            items.sort(key=lambda row: int(row.get("normal_point_index", row.get("record_index", 0))))
    if not rows:
        import numpy as np

        from lunarops.classes.observation.equations import EquationBatch

        return {
            source: batch[np.argsort(batch.observation_id, kind="stable")]
            for source, chunks in items_by_source.items()
            for batch in (EquationBatch.concatenate(chunks),)
        }
    return items_by_source


def mpi_prediction_rows(
    runtime: MpiRuntime,
    spec: dict,
    epochs: Sequence,
    *,
    step_seconds: float,
    station: str,
    reflector: str,
    criteria: dict,
    meteorology: dict,
    utc_offset_hours: float,
    chunksize: int = 8,
    progress_desc: str = "LLR prediction",
    quiet: bool = False,
) -> list[dict[str, object]]:
    spec["prediction"] = {
        "station": station, "reflector": reflector, "criteria": criteria,
        "meteorology": meteorology, "utcOffsetHours": utc_offset_hours,
    }
    spec_id = runtime.broadcast_spec("prediction", spec)
    chunk = int(chunksize)
    epoch_values = list(epochs)
    payloads = []
    for start in range(0, len(epoch_values), chunk):
        selected = epoch_values[start : start + chunk]
        payloads.append(
            {
                "specId": spec_id,
                "startEpoch": epoch_values[0],
                "startIndex": start,
                "recordCount": len(selected),
                "stepSeconds": step_seconds,
            }
        )
    results = runtime.map_tasks(
        "prediction",
        payloads,
        progress=None if quiet else progress_desc,
    )
    rows: list[tuple[int, dict[str, object]]] = []
    for result in results:
        for item in result["items"]:
            rows.append((int(item["index"]), item["row"]))
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows]
