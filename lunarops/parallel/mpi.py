"""MPI master-worker backend (port of the v24 ``run_*_mpi.py`` scripts).

Launch any config under MPI — nothing else changes::

    mpirun -n 16 python -m lunarops run config.yml --mpi
    srun python -m lunarops run config.yml --mpi        # SLURM, multi-node

Architecture (identical to v24, generalized):

rank 0
    * Parses the config, loads MINI files once, runs the program sequence.
    * Programs that support MPI dispatch dynamic chunks of already parsed
      ``NptRecord`` objects to worker ranks and gather results.
rank 1..N
    * Enter a lightweight generic worker loop before the program registry is
      imported.  Rank 0 broadcasts each immutable observation spec once
      (resolved configs, catalogs, parsed EOP arrays); later tasks carry only
      ``specId``.  Before task dispatch, rank 0 explicitly initializes every
      worker and waits for a ready response.  Each worker opens its own
      process-local CALCEPH handle once per spec and caches the resulting
      processor; initialization time is excluded from observation throughput.

Per-iteration state (updated reflector positions from the Gauss-Newton loop)
travels with every task as ``catalogState``, so relinearization works exactly
as in the serial path.

Task kinds:

``observation_equations``
    chunk of NptRecords -> typed equations for estimation or rows for residual
    output.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Sequence

from lunarops.parallel.observation_spec import (
    apply_catalog_state as _apply_catalog_state,
    build_worker_processor,
    make_observation_spec as make_observation_spec,
    snapshot_catalog_state as snapshot_catalog_state,
)

TAG_TASK = 101
TAG_RESULT = 102
TAG_STOP = 103
TAG_BROADCAST_SPEC = 104
TAG_INITIALIZE_SPEC = 105
TAG_READY = 106


def mpi_comm_world():
    """Import mpi4py lazily; raise a helpful error outside an MPI runtime."""
    try:
        from mpi4py import MPI  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "--mpi requires mpi4py and an MPI runtime; launch with "
            "mpirun/mpiexec/srun (e.g. 'srun python -m lunarops run cfg.yml --mpi')."
        ) from exc
    return MPI


def _processor_for_task(cache: dict, spec: dict):
    """Return one cached processor per observation spec on each worker rank.

    The v31 worker rebuilt the light-time processor for every tiny MPI chunk.
    Heavy CALCEPH/EOP objects were cached, but the processor graph was still
    reconstructed for every task and then closed immediately.
    Keep the processor warm for the lifetime of the worker, matching the MPI
    architecture documented at the top of this module.
    """
    processor_key = ("processor", spec["specId"])
    if processor_key not in cache:
        shared_class_cache = cache.setdefault("sharedClassCache", {})
        context, processor = build_worker_processor(spec, shared_class_cache)
        cache[("context", spec["specId"])] = context
        cache[processor_key] = processor
    return cache[processor_key]


def _close_worker_contexts(cache: dict) -> None:
    contexts = [value for key, value in cache.items() if isinstance(key, tuple) and key and key[0] == "context"]
    for context in contexts:
        context.close()
    shared_class_cache = cache.get("sharedClassCache")
    if isinstance(shared_class_cache, dict):
        from lunarops.resource_lifecycle import close_resources

        close_resources(shared_class_cache.values(), owner="mpi-worker-shared-cache")
        shared_class_cache.clear()


def _initialized_processor_for_task(cache: dict, spec: dict):
    """Return a processor created by the explicit initialization phase."""
    processor_key = ("processor", spec["specId"])
    try:
        return cache[processor_key]
    except KeyError:
        raise RuntimeError(
            f"MPI observation processor {spec['specId']!r} was not initialized before observation task dispatch."
        ) from None


# ---------------------------------------------------------------------------
# task handlers (executed on worker ranks; rank 0 falls back to them serially)
# ---------------------------------------------------------------------------


def _observation_spec_for_payload(payload: dict, cache: dict) -> dict:
    spec_id = str(payload["specId"])
    try:
        return cache[("observationSpec", spec_id)]
    except KeyError:
        raise RuntimeError(f"MPI observation spec {spec_id!r} was not broadcast before task dispatch.") from None


def _handle_observation_equations(payload: dict, cache: dict):
    """NptRecord chunk -> typed equations or lightweight table rows."""
    from lunarops.classes.observation.normal_points import NptDataset
    from lunarops.classes.observation import (
        ObservationResultDetail,
        ObservationProcessingOptions,
    )

    spec = _observation_spec_for_payload(payload, cache)
    processor = _initialized_processor_for_task(cache, spec)
    _apply_catalog_state(processor, payload.get("catalogState"))
    options = ObservationProcessingOptions(**payload["options"]).with_progress(
        None,
        enabled=False,
    )
    records = list(payload["records"])
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
    if payload.get("returnRows"):
        level = ObservationResultDetail.parse(payload.get("outputLevel", "standard"))
        response["items"] = processor.rows(local, options=options, detail=level)
    else:
        response["items"] = processor.equations(local, options=options)
    return response


TASK_HANDLERS: Dict[str, Callable[[dict, dict], object]] = {
    "observation_equations": _handle_observation_equations,
}


# ---------------------------------------------------------------------------
# runtime: generic dynamic master-worker scheduler
# ---------------------------------------------------------------------------


class MpiRuntime:
    """One communicator, one worker loop, one dynamic scheduler.

    Programs never talk MPI directly; they call the observation collection helpers, which
    routes through :meth:`map_tasks`.  Workers stay alive across programs and
    Gauss-Newton iterations, keeping their CALCEPH/EOP model caches warm.
    """

    def __init__(self, comm=None) -> None:
        self._MPI = mpi_comm_world()
        self.comm = comm if comm is not None else self._MPI.COMM_WORLD
        self._prepared_spec_ids: set[str] = set()
        self._initialized_spec_ids: set[str] = set()
        self._serial_cache: dict = {}

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        return self.comm.Get_size()

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def has_workers(self) -> bool:
        return self.size > 1

    def prepare_observation_spec(self, spec: dict) -> bool:
        """Broadcast one immutable observation spec exactly once.

        Workers remain in the generic receive loop.  A small point-to-point
        control message moves every rank into the same collective ``bcast``;
        the complete spec (catalogs plus parsed EOP arrays) is then transferred
        once and cached by ``specId``.  Individual tasks carry only the ID.
        """
        if not self.is_master:
            raise RuntimeError("Only rank 0 may broadcast observation specs.")
        spec_id = str(spec["specId"])
        if spec_id in self._prepared_spec_ids:
            return False

        if self.has_workers:
            command = {"kind": "observationSpec", "specId": spec_id}
            for worker in range(1, self.size):
                self.comm.send(command, dest=worker, tag=TAG_BROADCAST_SPEC)
            self.comm.bcast(spec, root=0)
        else:
            self._serial_cache[("observationSpec", spec_id)] = spec
        self._prepared_spec_ids.add(spec_id)
        return True

    def initialize_observation_workers(
        self,
        spec_id: str,
        *,
        quiet: bool = False,
    ) -> bool:
        """Construct and cache one observation processor on every worker.

        Initialization is a separate synchronization phase rather than a side
        effect of the first observation task.  Consequently CALCEPH opening,
        processor construction, and other one-time model setup are reported
        independently and are excluded from the timed observation throughput.
        """
        if not self.is_master:
            raise RuntimeError("Only rank 0 may initialize observation workers.")

        spec_id = str(spec_id)
        if spec_id not in self._prepared_spec_ids:
            raise RuntimeError(f"MPI observation spec {spec_id!r} must be broadcast before initialization.")
        if spec_id in self._initialized_spec_ids:
            return False

        if not self.has_workers:
            spec = self._serial_cache[("observationSpec", spec_id)]
            _processor_for_task(self._serial_cache, spec)
            self._initialized_spec_ids.add(spec_id)
            return True

        worker_ranks = list(range(1, self.size))
        if not quiet:
            print(
                f"[MPI] initializing {len(worker_ranks)} worker(s) (processor + process-local CALCEPH)...",
                flush=True,
            )

        command = {"kind": "initializeObservationSpec", "specId": spec_id}
        started = time.perf_counter()
        for worker in worker_ranks:
            self.comm.send(command, dest=worker, tag=TAG_INITIALIZE_SPEC)

        status = self._MPI.Status()
        pending = set(worker_ranks)
        failures: list[tuple[int, str]] = []
        worker_elapsed: dict[int, float] = {}
        while pending:
            is_error, response_spec_id, detail = self.comm.recv(
                source=self._MPI.ANY_SOURCE,
                tag=TAG_READY,
                status=status,
            )
            worker = status.Get_source()
            if worker not in pending:
                raise RuntimeError(f"Unexpected duplicate MPI initialization response from rank {worker}.")
            pending.remove(worker)
            if str(response_spec_id) != spec_id:
                failures.append(
                    (
                        worker,
                        "MPI initialization response spec ID mismatch: "
                        f"expected {spec_id!r}, received {response_spec_id!r}.",
                    )
                )
            elif is_error:
                failures.append((worker, str(detail)))
            elif isinstance(detail, dict):
                worker_elapsed[worker] = float(detail.get("elapsedSeconds", 0.0))

        elapsed = time.perf_counter() - started
        if failures:
            worker, remote_traceback = failures[0]
            extra = "" if len(failures) == 1 else f" ({len(failures)} workers failed)"
            raise RuntimeError(
                f"MPI worker rank {worker} failed while initializing observation "
                f"spec {spec_id!r}{extra}:\n{remote_traceback}"
            )

        self._initialized_spec_ids.add(spec_id)
        if not quiet:
            slowest = max(worker_elapsed.values(), default=0.0)
            print(
                f"[MPI] {len(worker_ranks)}/{len(worker_ranks)} worker(s) ready; "
                f"initialization {elapsed:.2f} s, slowest worker {slowest:.2f} s.",
                flush=True,
            )
        return True

    # -- worker side ---------------------------------------------------------
    def worker_loop(self) -> None:
        status = self._MPI.Status()
        cache: dict = {}
        try:
            while True:
                message = self.comm.recv(source=0, tag=self._MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == TAG_STOP:
                    break
                if tag == TAG_BROADCAST_SPEC:
                    command = message or {}
                    if command.get("kind") != "observationSpec":
                        raise RuntimeError(f"Rank {self.rank} received invalid broadcast command {command!r}.")
                    spec = self.comm.bcast(None, root=0)
                    spec_id = str(command["specId"])
                    if str(spec.get("specId")) != spec_id:
                        raise RuntimeError(
                            "MPI observation-spec broadcast ID mismatch: "
                            f"command={spec_id!r}, payload={spec.get('specId')!r}."
                        )
                    cache[("observationSpec", spec_id)] = spec
                    continue
                if tag == TAG_INITIALIZE_SPEC:
                    command = message or {}
                    spec_id = str(command.get("specId", ""))
                    if command.get("kind") != "initializeObservationSpec" or not spec_id:
                        self.comm.send(
                            (
                                True,
                                spec_id,
                                f"Rank {self.rank} received invalid initialization command {command!r}.",
                            ),
                            dest=0,
                            tag=TAG_READY,
                        )
                        continue
                    started = time.perf_counter()
                    try:
                        spec = cache[("observationSpec", spec_id)]
                        _processor_for_task(cache, spec)
                        self.comm.send(
                            (
                                False,
                                spec_id,
                                {"elapsedSeconds": time.perf_counter() - started},
                            ),
                            dest=0,
                            tag=TAG_READY,
                        )
                    except Exception:
                        self.comm.send(
                            (True, spec_id, traceback.format_exc()),
                            dest=0,
                            tag=TAG_READY,
                        )
                    continue
                if tag != TAG_TASK:
                    raise RuntimeError(f"Rank {self.rank} received unexpected MPI tag {tag}")
                kind, payload = message
                task_id = payload.get("taskId")
                try:
                    result = TASK_HANDLERS[kind](payload, cache)
                    self.comm.send((False, task_id, result), dest=0, tag=TAG_RESULT)
                except Exception:
                    self.comm.send((True, task_id, traceback.format_exc()), dest=0, tag=TAG_RESULT)
        finally:
            _close_worker_contexts(cache)

    # -- master side -----------------------------------------------------------
    def map_tasks(
        self,
        kind: str,
        payloads: Sequence[dict],
        *,
        progress_desc: Optional[str] = None,
        progress_total: Optional[int] = None,
        quiet: bool = False,
    ) -> List[object]:
        """Dynamically schedule payloads over worker ranks; return results in
        task order.  Worker exceptions re-raise here with the remote traceback.

        With no workers (size 1) tasks run serially in-process — the same
        fallback the v24 scripts implemented.
        """
        payloads = [dict(p, taskId=i) for i, p in enumerate(payloads)]
        n_tasks = len(payloads)
        if n_tasks == 0:
            return []

        if not self.has_workers:
            return [TASK_HANDLERS[kind](p, self._serial_cache) for p in payloads]

        status = self._MPI.Status()
        results: Dict[int, object] = {}
        worker_ranks = list(range(1, self.size))
        next_task = 0
        completed = 0
        completed_units = 0

        def send_next(worker_rank: int) -> None:
            nonlocal next_task
            if next_task >= n_tasks:
                return
            self.comm.send((kind, payloads[next_task]), dest=worker_rank, tag=TAG_TASK)
            next_task += 1

        desc = progress_desc or kind
        total = progress_total if progress_total is not None else n_tasks
        if not quiet:
            print(
                f"\r{desc}: 0/{total} (starting, tasks 0/{n_tasks}, ranks {self.size})",
                end="",
                flush=True,
            )

        # Start formal work only after the explicit worker-initialization
        # barrier has completed. Prime workers before starting throughput
        # timing so initial task sends/serialization do not dilute the rate.
        for worker in worker_ranks[: min(len(worker_ranks), n_tasks)]:
            send_next(worker)

        rate_started = time.perf_counter()
        last_report = rate_started
        last_report_units = 0
        while completed < n_tasks:
            is_error, task_id, result = self.comm.recv(source=self._MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
            worker = status.Get_source()
            if is_error:
                raise RuntimeError(f"MPI worker rank {worker} failed on {kind} task {task_id}:\n{result}")
            results[int(task_id)] = result
            completed += 1
            if isinstance(result, dict):
                completed_units += int(result.get("nRecords", 1))

            now = time.perf_counter()
            if not quiet and (now - last_report >= 2.0 or completed == n_tasks):
                elapsed = max(1.0e-9, now - rate_started)
                done = completed_units if progress_total is not None else completed
                window_elapsed = max(1.0e-9, now - last_report)
                window_done = done - last_report_units
                avg_rate = done / elapsed
                recent_rate = window_done / window_elapsed
                print(
                    f"\r{desc}: {done}/{total} "
                    f"({avg_rate:.2f}/s avg, {recent_rate:.2f}/s recent, "
                    f"active {elapsed:.1f}s, tasks {completed}/{n_tasks}, "
                    f"ranks {self.size})",
                    end="" if completed < n_tasks else "\n",
                    flush=True,
                )
                last_report = now
                last_report_units = done

            send_next(worker)

        return [results[i] for i in range(n_tasks)]

    def shutdown(self) -> None:
        if not self.is_master:
            return
        if self.has_workers:
            for worker in range(1, self.size):
                self.comm.send(None, dest=worker, tag=TAG_STOP)
        _close_worker_contexts(self._serial_cache)
        self._serial_cache.clear()
        self._prepared_spec_ids.clear()
        self._initialized_spec_ids.clear()


# ---------------------------------------------------------------------------
# program-facing helpers
# ---------------------------------------------------------------------------


def chunk_dataset_tasks(datasets, chunksize: int) -> List[dict]:
    """Chunk already parsed NptRecords of every source (v24 ``_make_tasks``)."""
    chunk = max(1, int(chunksize))
    tasks: List[dict] = []
    for source_name, dataset in datasets.items():
        records = list(dataset.records)
        for start in range(0, len(records), chunk):
            tasks.append(
                {
                    "sourceName": str(source_name),
                    "startIndex": start,
                    "records": records[start : start + chunk],
                }
            )
    return tasks


def _observation_task_payloads(
    spec_id: str,
    datasets,
    options,
    *,
    chunksize: int,
    catalog_state: Optional[dict],
    return_rows: bool = False,
    output_level: str = "standard",
) -> list[dict]:
    options_dict = asdict(options)
    return [
        dict(
            task,
            specId=str(spec_id),
            options=options_dict,
            catalogState=catalog_state,
            returnRows=bool(return_rows),
            outputLevel=str(output_level),
        )
        for task in chunk_dataset_tasks(datasets, chunksize)
    ]


def _prepare_and_initialize_spec(
    runtime: MpiRuntime,
    spec: dict,
    *,
    quiet: bool,
) -> None:
    prepared = runtime.prepare_observation_spec(spec)
    if prepared and not quiet:
        earth_rotation = (spec.get("sharedResources") or {}).get("earthRotation") or {}
        sample_count = len(earth_rotation.get("mjdUtc", ()))
        detail = f", EOP samples={sample_count}" if sample_count else ""
        print(
            f"[MPI] observation spec broadcast to {runtime.size - 1} worker(s){detail}.",
            flush=True,
        )
    runtime.initialize_observation_workers(spec["specId"], quiet=quiet)


def _collect_observations(
    runtime: MpiRuntime,
    spec: dict,
    datasets,
    options,
    *,
    chunksize: int = 8,
    catalog_state: Optional[dict] = None,
    return_rows: bool,
    output_level: str,
    progress_desc: str = "O-C normal points",
    quiet: bool = False,
) -> Dict[str, list]:
    _prepare_and_initialize_spec(runtime, spec, quiet=quiet)
    tasks = _observation_task_payloads(
        spec["specId"],
        datasets,
        options,
        chunksize=chunksize,
        catalog_state=catalog_state,
        return_rows=return_rows,
        output_level=output_level,
    )
    total = sum(len(dataset.records) for dataset in datasets.values())
    results = runtime.map_tasks(
        "observation_equations",
        tasks,
        progress_desc=progress_desc,
        progress_total=total,
        quiet=quiet,
    )
    items_by_source: Dict[str, list] = {str(name): [] for name in datasets}
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("Observation worker returned a non-mapping result.")
        source_name = result.get("sourceName")
        items = result.get("items")
        if not isinstance(source_name, str) or not isinstance(items, list):
            raise TypeError("Observation worker result has an invalid payload shape.")
        items_by_source[source_name].extend(items)
    for items in items_by_source.values():
        if return_rows:
            items.sort(key=lambda row: int(row.get("normal_point_index", row.get("record_index", 0))))
        else:
            items.sort(key=lambda equation: equation.observation_id)
    return items_by_source


def mpi_observation_equations(
    runtime: MpiRuntime,
    spec: dict,
    datasets,
    options,
    *,
    chunksize: int = 8,
    catalog_state: Optional[dict] = None,
    progress_desc: str = "O-C normal points",
    quiet: bool = False,
) -> Dict[str, list]:
    """Compute typed observation equations for every dataset over MPI."""
    return _collect_observations(
        runtime,
        spec,
        datasets,
        options,
        chunksize=chunksize,
        catalog_state=catalog_state,
        return_rows=False,
        output_level="standard",
        progress_desc=progress_desc,
        quiet=quiet,
    )


def mpi_observation_rows(
    runtime: MpiRuntime,
    spec: dict,
    datasets,
    options,
    *,
    output_level: str = "standard",
    chunksize: int = 8,
    catalog_state: Optional[dict] = None,
    progress_desc: str = "O-C normal points",
    quiet: bool = False,
) -> Dict[str, list[dict]]:
    """Compute lightweight O-C table rows over MPI.

    Residual-table production does not need to ship full typed equations,
    partial arrays, or Epoch instances back to rank 0. Adjustment workflows use
    the typed-equation path; table workflows request projected rows.
    """
    return _collect_observations(
        runtime,
        spec,
        datasets,
        options,
        chunksize=chunksize,
        catalog_state=catalog_state,
        return_rows=True,
        output_level=output_level,
        progress_desc=progress_desc,
        quiet=quiet,
    )
