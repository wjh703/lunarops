"""Master-worker MPI execution.

Rank 0 runs the program sequence; ranks 1..N run ``MpiRuntime.worker_loop``. The master first broadcasts a
spec (an observation model, a predictor or lunar-dynamics epoch inputs), from which every worker builds
its model once, and then streams small tasks that refer to the spec by ID: chunks of normal points,
chunks of prediction epochs or batches of integration epochs. Results are returned to the master in task
order. The master side shares its protocol with ``ProcessRuntime`` (``TaskRuntimeBase``); the workers keep
their models in a ``WorkerState`` and serve specs and tasks with ``initialize_spec`` and ``run_task`` of
:mod:`lunarops.parallel.tasks`, as ``ProcessRuntime`` workers do.

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
import traceback
from typing import Any, NoReturn

from lunarops import console
from lunarops.parallel.runtime import TaskRuntimeBase, is_interrupt
from lunarops.parallel.tasks import FLUSH_TASK, SPEC_LABELS, WorkerState, initialize_spec, run_task

# MPI message tags: task from the master, result from a worker, stop request.
TASK = 101
RESULT = 102
STOP = 103
#: Completed sends are pruned from the send records once there are more than this many records per rank.
SEND_RECORDS_PER_RANK = 4


def _import_mpi():
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


class MpiRuntime(TaskRuntimeBase):
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
        self._MPI = _import_mpi()
        self.comm = comm if comm is not None else self._MPI.COMM_WORLD
        self._spec_ids = {}
        self._stream = None
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

    def _prepare_broadcast(self) -> None:
        self._settle()

    def _broadcast(self, kind: str, spec_id: int, spec: dict) -> None:
        """Send a spec to every worker and wait until all have built its model.

        Any exception or interrupt between the first spec message and the end of the gather aborts the job:
        some workers may then wait in the collective broadcast or gather, which nothing else can release.
        """
        label = SPEC_LABELS.get(kind, kind)
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

    # A stream: each worker holds up to two tasks at a time and receives the next one when it returns a result.
    # A stream stopped by an error or closed early settles the workers before it ends; one stopped by an
    # interrupt (``is_interrupt``) does not wait for them, and the next stream, broadcast or shutdown settles
    # them.

    def _begin_stream(self) -> None:
        self._settle()
        # Until the stream completes, the workers may hold tasks or results of it; a stopped stream is settled
        # with a flush, since its counters may have missed the message that was being received.
        self._unsettled = True

    def _task_workers(self) -> range:
        return range(1, self.size)

    def _send_task(self, worker: int, kind: str, payload: dict) -> None:
        self._isend((kind, payload), worker)
        if len(self._sends) > SEND_RECORDS_PER_RANK * self.size:
            self._sends[:] = [send for send in self._sends if send[1] is None or not send[1].Test()]

    def _receive_result(self, kind: str) -> tuple[int, int, Any]:
        status = self._MPI.Status()
        is_error, task_id, result = self.comm.recv(source=self._MPI.ANY_SOURCE, tag=RESULT, status=status)
        worker = status.Get_source()
        if is_error:
            raise RuntimeError(f"MPI worker rank {worker} failed on {kind} task {task_id}:\n{result}")
        return worker, task_id, result

    def _stop_stream(self, exc: BaseException, kind: str) -> None:
        if not is_interrupt(exc):
            self._settle()

    def _end_stream(self) -> None:
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
            token = (FLUSH_TASK, self._flushes)
            for worker in range(1, self.size):
                self._isend((FLUSH_TASK, {"taskId": token}), worker)
            waiting = set(range(1, self.size))
            status = self._MPI.Status()
            while waiting:
                _, task_id, _ = self.comm.recv(source=self._MPI.ANY_SOURCE, tag=RESULT, status=status)
                if task_id == token:
                    waiting.discard(status.Get_source())
            self._unsettled = False
        self._release_sends()

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
                        # A failure is forwarded once the collective gather completes.
                        self.comm.gather(initialize_spec(state, spec_kind, spec_id, spec), root=0)
                    else:
                        result = run_task(kind, payload, state)
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
