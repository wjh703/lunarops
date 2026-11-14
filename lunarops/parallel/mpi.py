"""Master-worker MPI execution.

Rank 0 runs the program sequence; ranks 1..N run ``MpiRuntime.worker_loop``. The master first broadcasts a
spec (an observation model, a predictor or lunar-dynamics epoch inputs), from which every worker builds
its model once, and then streams small tasks that refer to the spec by ID: chunks of normal points,
chunks of prediction epochs or batches of integration epochs. Results are returned to the master in task
order. The workers keep their models in a ``WorkerState`` and look task handlers up by kind in
``TASK_HANDLERS`` (both in :mod:`lunarops.parallel.tasks`, shared with ``ProcessRuntime``).

Errors and interrupts on rank 0 follow one rule (see ``MpiRuntime.shutdown``): an ordinary exception, a
worker task failure included, leaves the workers in a state that a flush can settle, and the job ends with a
stop message to every worker; an interrupt (``is_interrupt``), or any failure while the workers cannot be
settled (inside a spec broadcast, or while shutting down), aborts the whole job with ``MPI_Abort``. Workers
ignore SIGINT, so that a launcher that signals every rank leaves the decision to rank 0.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
import traceback
from collections.abc import Generator, Iterator, Sequence
from dataclasses import asdict
from typing import Any, NoReturn

from lunarops import console
from lunarops.parallel.runtime import TaskRuntime, is_interrupt
from lunarops.parallel.tasks import FLUSH_TASK as _FLUSH
from lunarops.parallel.tasks import SPEC_LABELS as _SPEC_LABELS
from lunarops.parallel.tasks import TASK_HANDLERS, WorkerState

# MPI message tags: task from the master, result from a worker, stop request.
TASK = 101
RESULT = 102
STOP = 103


def mpi_comm_world():
    """Return the ``mpi4py.MPI`` module.

    Returns
    -------
    module
        ``mpi4py.MPI``.

    Raises
    ------
    SystemExit
        If ``mpi4py`` or the MPI library cannot be imported.
    """
    try:
        from mpi4py import MPI
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "--mpi requires mpi4py and an MPI runtime; launch with "
            "mpirun/mpiexec/srun (e.g. 'srun python -m lunarops run cfg.yml --mpi')."
        ) from exc
    return MPI


class MpiRuntime:
    """MPI communicator wrapper for master-worker task execution.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        Communicator; ``MPI.COMM_WORLD`` when None.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        Communicator in use.
    name : str
        Runtime name in run metadata, ``mpi``.

    Raises
    ------
    RuntimeError
        If the communicator has fewer than two ranks.
    """

    name = "mpi"

    def __init__(self, comm=None) -> None:
        self._MPI = mpi_comm_world()
        self.comm = comm if comm is not None else self._MPI.COMM_WORLD
        self._spec_ids: dict[str, int] = {}
        self._stream: Generator | None = None
        self._unsettled = False
        self._flushes = 0
        # [pickled message, request or None] of every task send not known to be complete (see _isend).
        self._sends: list[list] = []
        if self.size < 2:
            raise RuntimeError("Explicit MPI execution requires at least two ranks (one worker).")

    @property
    def rank(self) -> int:
        """Rank of this process."""
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        """Number of ranks, master included."""
        return self.comm.Get_size()

    @property
    def is_master(self) -> bool:
        """Whether this process is rank 0."""
        return self.rank == 0

    @property
    def workers(self) -> int:
        """Number of worker ranks."""
        return self.size - 1

    def describe(self) -> str:
        """Return the execution mode shown in the run header.

        Returns
        -------
        str
            For example ``MPI, 4 ranks (3 workers)``.
        """
        return f"MPI, {self.size} ranks ({self.workers} workers)"

    def broadcast_spec(self, kind: str, spec: dict) -> int:
        """Send a spec to every worker and wait until all have built its model.

        A spec whose ``specId`` was already broadcast is not sent again.

        Parameters
        ----------
        kind : str
            Spec kind: ``observationSpec``, ``prediction`` or ``epochInputs``.
        spec : dict
            Spec contents with a ``specId`` entry.

        Returns
        -------
        int
            Spec ID used in the task payloads.

        Raises
        ------
        RuntimeError
            If a task stream is open, or a worker fails to build the model.

        Notes
        -----
        Any exception or interrupt between the first spec message and the end of the gather aborts the job:
        some workers may then wait in the collective broadcast or gather, which nothing else can release.
        """
        if self._stream is not None:
            raise RuntimeError("broadcast_spec while a task stream is open")
        self._settle()
        key = spec["specId"]
        if key in self._spec_ids:
            return self._spec_ids[key]
        spec_id = len(self._spec_ids)
        label = _SPEC_LABELS.get(kind, kind)
        try:
            for worker in range(1, self.size):
                self.comm.send(("spec", (kind, spec_id)), dest=worker, tag=TASK)
            self.comm.bcast(spec, root=0)
            timings = self.comm.gather((0.0, ""), root=0)
        except BaseException as exc:  # noqa: BLE001 -- workers may wait in the collective: only an abort ends it
            self.abort(f"{type(exc).__name__} while broadcasting the {label}", exc)
        for worker, (_, error) in enumerate(timings[1:], start=1):
            if error:
                raise RuntimeError(f"MPI worker rank {worker} failed initializing {kind}:\n{error}")
        console.info(f"{label} ready on {self.size - 1} MPI workers ({max(t[0] for t in timings):.2f} s)")
        self._spec_ids[key] = spec_id
        return spec_id

    def stream_tasks(
        self,
        kind: str,
        payloads: Sequence[dict],
        *,
        progress: str | None = None,
        unit: str = "tasks",
        sizes: Sequence[int] | None = None,
    ) -> Iterator[Any]:
        """Distribute tasks over the workers and return an iterator over their results in task order.

        Each worker holds up to two tasks at a time and receives the next one when it returns a result. Only
        one iterator can be open at a time, and no spec can be broadcast while it is: exhaust or close it
        first. An iterator stopped by an error or closed early settles the workers before it ends; one stopped
        by an interrupt (``is_interrupt``) does not wait for them, and the next stream, broadcast or shutdown
        settles them.

        Parameters
        ----------
        kind : str
            Task kind, a key of ``TASK_HANDLERS``.
        payloads : sequence of dict
            Task payloads; each is sent with its position as ``taskId``.
        progress : str, optional
            Progress label; no progress is reported when None.
        unit : str, optional
            Plural name of the items counted by the progress.
        sizes : sequence of int, optional
            Number of items in each task; each task counts as one item when None.

        Returns
        -------
        iterator
            Results in the order of ``payloads``.

        Raises
        ------
        RuntimeError
            If another task stream is open (when called, or when the iterator starts), or, while iterating,
            if a worker reports a task failure.
        """
        if self._stream is not None:
            raise RuntimeError("stream_tasks while another task stream is open")

        def stream():
            if self._stream is not None:
                raise RuntimeError("stream_tasks while another task stream is open")
            self._stream = iterator
            try:
                yield from self._stream_tasks(kind, payloads, progress=progress, unit=unit, sizes=sizes)
            finally:
                if self._stream is iterator:
                    self._stream = None

        iterator = stream()
        return iterator

    def _stream_tasks(
        self, kind: str, payloads: Sequence[dict], *, progress: str | None, unit: str, sizes: Sequence[int] | None
    ) -> Iterator[Any]:
        self._settle()
        payloads = [dict(p, taskId=i) for i, p in enumerate(payloads)]
        counts = [1] * len(payloads) if sizes is None else [int(size) for size in sizes]
        status = self._MPI.Status()
        results: dict[int, Any] = {}
        next_task = 0

        def send_next(worker_rank: int) -> None:
            nonlocal next_task
            if next_task < len(payloads):
                self._isend((kind, payloads[next_task]), worker_rank)
                next_task += 1
                if len(self._sends) > 4 * self.size:
                    self._sends[:] = [send for send in self._sends if send[1] is None or not send[1].Test()]

        with console.progress(total=sum(counts), desc=progress or "", unit=unit, disable=progress is None) as bar:

            def receive() -> None:
                is_error, task_id, result = self.comm.recv(source=self._MPI.ANY_SOURCE, tag=RESULT, status=status)
                worker = status.Get_source()
                if is_error:
                    raise RuntimeError(f"MPI worker rank {worker} failed on {kind} task {task_id}:\n{result}")
                results[task_id] = result
                bar.update(counts[task_id])
                send_next(worker)

            # Until the stream completes, the workers may hold tasks or results of it; a stopped stream is
            # settled with a flush, since its counters may have missed the message that was being received.
            self._unsettled = True
            try:
                for _ in range(2):
                    for worker in range(1, self.size):
                        send_next(worker)
                for wanted in range(len(payloads)):
                    while wanted not in results:
                        receive()
                    yield results.pop(wanted)
            except BaseException as exc:
                next_task = len(payloads)
                if not is_interrupt(exc):
                    self._settle()
                raise
        # Every task was answered, so nothing of this stream is left on the workers.
        self._unsettled = False
        self._release_sends()

    def _isend(self, message, dest: int) -> None:
        """Start sending a task message to a worker.

        The message is pickled here and recorded in ``_sends`` before the send starts, so that the buffer
        stays referenced until the send is known to be complete. An interrupt raised right after ``Isend``
        returns loses its request (mpi4py frees a dropped request with ``MPI_Request_free``, which lets the
        send complete), but not the buffer, which is released once the runtime is settled. The receiving
        ``comm.recv`` unpickles the message as if ``comm.isend`` had sent it.

        Parameters
        ----------
        message : object
            Picklable message.
        dest : int
            Worker rank.
        """
        send = [self._MPI.pickle.dumps(message), None]
        self._sends.append(send)
        send[1] = self.comm.Isend([send[0], self._MPI.BYTE], dest=dest, tag=TASK)

    def _release_sends(self) -> None:
        """Complete the recorded task sends and release their buffers.

        Called only while the runtime is settled: every worker has then received, in order, every message
        sent to it, so the buffer of a send whose request was lost is no longer read. An interrupted wait
        keeps the records for the next call.
        """
        self._MPI.Request.Waitall([request for _, request in self._sends if request is not None])
        self._sends.clear()

    def _settle(self) -> None:
        """Discard everything a stopped stream left on the workers.

        A stream stopped by an error or an interrupt may leave tasks queued on the workers and results in
        flight, and an interrupt between a receive or a send and its bookkeeping makes a count of them
        unreliable. Instead, every worker receives a flush task, behind any task it still holds, and the
        results are discarded until each worker has answered it: messages between two ranks keep their order.
        An interrupted flush leaves the runtime unsettled, and the next stream, broadcast or shutdown flushes
        again with a new token. The pending task sends are completed once the runtime is settled.
        """
        if self._unsettled:
            self._flushes += 1
            token = (_FLUSH, self._flushes)
            for worker in range(1, self.size):
                self._isend((_FLUSH, {"taskId": token}), worker)
            waiting = set(range(1, self.size))
            status = self._MPI.Status()
            while waiting:
                _, task_id, _ = self.comm.recv(source=self._MPI.ANY_SOURCE, tag=RESULT, status=status)
                if task_id == token:
                    waiting.discard(status.Get_source())
            self._unsettled = False
        self._release_sends()

    def map_tasks(
        self,
        kind: str,
        payloads: Sequence[dict],
        *,
        progress: str | None = None,
        unit: str = "tasks",
        sizes: Sequence[int] | None = None,
    ) -> list[Any]:
        """Run tasks on the workers and return all results in task order (see ``stream_tasks``)."""
        return list(self.stream_tasks(kind, payloads, progress=progress, unit=unit, sizes=sizes))

    def worker_loop(self) -> None:
        """Serve spec and task messages from rank 0 until a stop message arrives.

        Task failures are sent back to the master with their traceback; a malformed message, or any other
        exception that would leave the loop before the stop message, aborts the communicator. SIGINT is
        ignored while the loop runs (when it runs in the main thread): a launcher that forwards Ctrl-C to
        every rank leaves it to rank 0 to stop or abort the job.
        """
        status = self._MPI.Status()
        state = WorkerState()
        main_thread = threading.current_thread() is threading.main_thread()
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN) if main_thread else None
        try:
            while True:
                message = self.comm.recv(source=0, tag=self._MPI.ANY_TAG, status=status)
                if status.Get_tag() == STOP:
                    break
                try:
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
                        result = TASK_HANDLERS[kind](payload, state)
                        self.comm.send((False, payload["taskId"], result), dest=0, tag=RESULT)
                except Exception:  # noqa: BLE001 -- forward remote protocol/task failures or abort the communicator
                    error = traceback.format_exc()
                    try:
                        task_id = message[1]["taskId"]
                    except IndexError, KeyError, TypeError:
                        self.abort(f"malformed message from rank 0:\n{error}")
                    else:
                        self.comm.send((True, task_id, error), dest=0, tag=RESULT)
        except BaseException as exc:  # noqa: BLE001 -- leaving without a stop message would hang rank 0
            self.abort(f"{type(exc).__name__} in the worker loop", exc)
        finally:
            state.close()
            if main_thread:
                signal.signal(signal.SIGINT, previous_handler)

    def abort(self, reason: str, exc: BaseException | None = None) -> NoReturn:
        """Report why on standard error and abort every rank with ``MPI_Abort``.

        SIGINT is ignored first, so that a further Ctrl-C cannot stop the abort itself.

        Parameters
        ----------
        reason : str
            Short description of the failure.
        exc : BaseException, optional
            Exception whose traceback is printed (not for ``KeyboardInterrupt``).
        """
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            if exc is not None and not isinstance(exc, KeyboardInterrupt):
                traceback.print_exception(exc, file=sys.stderr)
            print(f"MPI rank {self.rank}: {reason}; aborting the MPI job.", file=sys.stderr, flush=True)
        except BaseException:  # noqa: BLE001,S110 -- best-effort message; the job is aborted regardless
            pass
        self.comm.Abort(1)
        raise SystemExit(1)  # not reached: MPI_Abort does not return

    def shutdown(self, *, interrupted: bool = False) -> None:
        """Stop the workers at the end of a run: settle them and send a stop message, or abort the job.

        This is the rank-0 end of the interrupt and abort rule:

        - After success or an ordinary exception (``is_interrupt`` is false, worker task failures included),
          an open task stream is closed, the workers are settled with a flush and every worker receives a
          stop message, so that all ranks reach ``MPI_Finalize``.
        - With ``interrupted`` (the run was stopped by ``KeyboardInterrupt`` or another interrupt), the job
          is aborted at once, without waiting for the tasks the workers hold.
        - Any exception or interrupt while closing, settling or stopping (a second Ctrl-C, a failed worker)
          aborts the job, since a worker that misses its stop message would hang it in ``MPI_Finalize``.

        Parameters
        ----------
        interrupted : bool, optional
            Whether the run was interrupted.
        """
        if interrupted:
            self.abort("interrupted")
        try:
            if self._stream is not None:
                self._stream.close()
            self._settle()
            for worker in range(1, self.size):
                self.comm.send(None, dest=worker, tag=STOP)
        except BaseException as exc:  # noqa: BLE001 -- a worker without a stop message hangs MPI_Finalize
            self.abort(f"{type(exc).__name__} while stopping the MPI workers", exc)
        self._spec_ids.clear()


def chunk_dataset_tasks(datasets, chunksize: int) -> list[dict]:
    """Split datasets into tasks of at most ``chunksize`` consecutive normal points.

    Parameters
    ----------
    datasets : dict of str to NptDataset
        Datasets keyed by source name.
    chunksize : int
        Maximum number of normal points per task.

    Returns
    -------
    list of dict
        Tasks with ``sourceName``, ``startIndex`` and ``recordCount``.
    """
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
    runtime: TaskRuntime,
    spec: dict,
    datasets,
    options,
    *,
    chunksize: int = 8,
    catalog_state: dict | None = None,
    rows: bool,
    output_level: str = "standard",
    progress_desc: str = "residuals",
    quiet: bool = False,
) -> dict[str, list]:
    """Evaluate normal points on the workers of a task runtime.

    The datasets are sent once with the spec (if the spec has none yet); each task then names a chunk of
    normal points. ``catalog_state`` sends the current reflector and station states with every task.

    Parameters
    ----------
    runtime : TaskRuntime
        Task runtime of the master.
    spec : dict
        Observation spec from ``make_observation_spec``.
    datasets : dict of str to NptDataset
        Datasets keyed by source name.
    options : ObservationOptions
        Observation options.
    chunksize : int, optional
        Normal points per task. Default 8.
    catalog_state : dict, optional
        Snapshot from ``snapshot_catalog_state`` applied by the workers before evaluating.
    rows : bool
        Return result-table rows (True) or observation equations (False).
    output_level : str, optional
        Column set of the rows: ``standard`` or ``full``.
    progress_desc : str, optional
        Progress label.
    quiet : bool, optional
        Report no progress.

    Returns
    -------
    dict of str to list or EquationBatch
        Per source: rows sorted by normal-point index, or one ``EquationBatch`` sorted by observation ID.
    """
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
        unit="normal points",
        sizes=[task["recordCount"] for task in tasks],
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
    runtime: TaskRuntime,
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
    progress_desc: str = "prediction",
    quiet: bool = False,
) -> list[dict[str, object]]:
    """Evaluate prediction rows on the workers of a task runtime.

    The epochs must form a grid ``epochs[0] + k * step_seconds``: each task sends only its start index and
    count, and the workers rebuild the epochs from the first epoch and the step.

    Parameters
    ----------
    runtime : TaskRuntime
        Task runtime of the master.
    spec : dict
        Observation spec from ``make_observation_spec``; the prediction settings are added to it.
    epochs : sequence of Epoch
        UTC grid epochs.
    step_seconds : float
        Grid spacing, in seconds.
    station : str
        Station catalog key.
    reflector : str
        Reflector catalog key.
    criteria : dict
        Fields of ``PredictionCriteria``.
    meteorology : dict
        Fields of ``PredictionMeteorology``.
    utc_offset_hours : float
        Civil-time offset from UTC of the local-time columns, in hours.
    chunksize : int, optional
        Epochs per task. Default 8.
    progress_desc : str, optional
        Progress label.
    quiet : bool, optional
        Report no progress.

    Returns
    -------
    list of dict
        Prediction rows in epoch order.
    """
    spec["prediction"] = {
        "station": station,
        "reflector": reflector,
        "criteria": criteria,
        "meteorology": meteorology,
        "utcOffsetHours": utc_offset_hours,
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
        unit="epochs",
        sizes=[payload["recordCount"] for payload in payloads],
    )
    rows: list[tuple[int, dict[str, object]]] = []
    for result in results:
        for item in result["items"]:
            rows.append((int(item["index"]), item["row"]))
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows]
