"""Interface shared by the task runtimes, and the process set-up of a parallel run.

A task runtime distributes model evaluations over worker processes: ``MpiRuntime`` over MPI ranks (``--mpi``)
and ``ProcessRuntime`` over local processes that it starts itself (``--workers N``). Programs see only the
``TaskRuntime`` interface: they broadcast a spec, from which every worker builds its model once, and then
stream small tasks that refer to the spec by ID. Both runtimes run the same task handlers
(:mod:`lunarops.parallel.tasks`), so a run gives the same results with either runtime and serially.

It also holds what the processes of a parallel run share about starting and stopping child processes:
``starting_children`` (child processes start with SIGINT ignored and one native thread), ``Terminated`` (SIGTERM
as an interrupt) and ``describe_exit``.

This module imports nothing heavy: ``configure_native_threads`` must run before NumPy is loaded.
"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

NATIVE_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
"""Environment variables that set the thread count of the native numerical libraries."""

QUEUED_TASKS_PER_WORKER = 2
"""Number of tasks a worker holds at a time: one in work and one queued."""

KILL_TIMEOUT_S = 5.0
"""Time a terminated child process has to exit before it is killed."""

_ORDERLY = (Exception, GeneratorExit, SystemExit)


class Terminated(BaseException):
    """SIGTERM turned into an exception by a process that must clean up before it exits (see ``raise_terminated``).

    It is an interrupt (``is_interrupt``): a task runtime that sees it stops its workers at once, also in the
    middle of a message.

    Parameters
    ----------
    signum : int, optional
        Signal received.

    Attributes
    ----------
    signum : int
        Signal received.
    """

    def __init__(self, signum: int = signal.SIGTERM) -> None:
        super().__init__(signum)
        self.signum = int(signum)

    @property
    def exit_code(self) -> int:
        """int: Exit code of a process ended by the signal, ``128 + signum`` (143 for SIGTERM)."""
        return 128 + self.signum


def raise_terminated(signum, frame) -> None:
    """Signal handler that raises ``Terminated`` once; the signal is ignored while the process cleans up.

    Parameters
    ----------
    signum : int
        Signal received.
    frame : frame or None
        Interrupted frame (unused).

    Raises
    ------
    Terminated
        Always.
    """
    signal.signal(signum, signal.SIG_IGN)
    raise Terminated(signum)


def configure_native_threads() -> None:
    """Limit the native numerical libraries to one thread per process.

    Every process of a parallel run is one busy CPU; BLAS or OpenMP thread pools on top of it would
    oversubscribe the CPUs. The variables are read when the libraries load, so this must run before NumPy is
    imported; processes started afterwards inherit them.
    """
    for name in NATIVE_THREAD_VARIABLES:
        os.environ[name] = "1"


@contextmanager
def starting_children() -> Iterator[None]:
    """Context in which to start the child processes of a parallel run (worker or backward-arc processes).

    Inside the block, SIGINT is ignored (in the main thread only: signal handlers can be set nowhere else) and
    the variables of ``NATIVE_THREAD_VARIABLES`` are set to 1; both are restored on leaving it. A child spawned
    inside therefore ignores SIGINT from its first instruction (an ignored signal stays ignored across
    ``exec``, and Python keeps it so), so that Ctrl-C in a terminal, which signals the whole process group,
    cannot interrupt its start-up; and its native numerical libraries load with one thread, whatever this
    process uses. Ctrl-C is lost while a child starts, which takes milliseconds.

    Yields
    ------
    None
        Inside the block, start the child processes.
    """
    saved = {name: os.environ.get(name) for name in NATIVE_THREAD_VARIABLES}
    # None: a handler set outside Python, which could not be restored.
    previous = signal.getsignal(signal.SIGINT) if threading.current_thread() is threading.main_thread() else None
    if previous is not None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for name in NATIVE_THREAD_VARIABLES:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def describe_exit(exitcode: int | None) -> str:
    """Describe how a child process ended, for error messages.

    Parameters
    ----------
    exitcode : int or None
        ``multiprocessing.Process.exitcode``: negative for a signal, None while the process runs.

    Returns
    -------
    str
        For example ``exited with code 3`` or ``was killed by SIGKILL``; ``closed its pipe`` for a process
        that still runs.
    """
    if exitcode is None:
        return "closed its pipe"
    if exitcode < 0:
        try:
            return f"was killed by {signal.Signals(-exitcode).name}"
        except ValueError:
            return f"was killed by signal {-exitcode}"
    return f"exited with code {exitcode}"


def is_interrupt(exc: BaseException) -> bool:
    """Return whether an exception interrupts a parallel run rather than ending it in an orderly way.

    Ordinary exceptions (``Exception``, which includes worker task failures), ``GeneratorExit`` (a task
    stream closed early) and ``SystemExit`` are orderly; any other ``BaseException``, ``KeyboardInterrupt``
    or ``Terminated`` in practice, is an interrupt, on which the runtime stops its workers without waiting
    for their tasks.

    Parameters
    ----------
    exc : BaseException
        Exception that stopped the run.

    Returns
    -------
    bool
        True for an interrupt.
    """
    return not isinstance(exc, _ORDERLY)


@runtime_checkable
class TaskRuntime(Protocol):
    """Master side of a runtime that evaluates tasks on worker processes.

    Attributes
    ----------
    name : str
        Runtime name recorded in run metadata: ``mpi`` or ``process``.
    """

    name: str

    @property
    def workers(self) -> int:
        """Number of worker processes."""
        ...

    def describe(self) -> str:
        """Return the execution mode shown in the run header, e.g. ``4 worker processes``."""
        ...

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
        """
        ...

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

        Only one iterator can be open at a time, and no spec can be broadcast while it is: exhaust or close
        it first.

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
        """
        ...

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
        ...

    def shutdown(self, *, interrupted: bool = False) -> None:
        """Stop the workers at the end of a run.

        Parameters
        ----------
        interrupted : bool, optional
            Whether the run was interrupted (``is_interrupt``); the workers are then stopped without waiting
            for the tasks they hold.
        """
        ...


class TaskRuntimeBase:
    """Master-side protocol shared by ``MpiRuntime`` and ``ProcessRuntime``.

    It holds what does not depend on the transport: the spec-ID cache of ``broadcast_spec``, the one-open-stream
    rule of ``stream_tasks``, ``map_tasks``, and the task loop of a stream (task IDs, progress, two tasks queued
    per worker, results yielded in task order). A runtime supplies the transport through the methods below;
    each keeps its own rules for errors, interrupts and stopped streams (``_stop_stream``).

    The subclass sets ``self._spec_ids = {}`` and ``self._stream = None`` in its constructor.
    """

    _spec_ids: dict[str, int]
    _stream: Generator | None

    # Transport of a runtime.

    def _check_usable(self) -> None:
        """Raise if the runtime can no longer run tasks; called by ``stream_tasks`` before a stream is created."""

    def _prepare_broadcast(self) -> None:
        """Make the workers ready for a spec broadcast; called before the spec cache is looked up."""
        raise NotImplementedError

    def _broadcast(self, kind: str, spec_id: int, spec: dict) -> None:
        """Send a spec to every worker and wait until all have built its model, or raise."""
        raise NotImplementedError

    def _begin_stream(self) -> None:
        """Make the workers ready for a task stream; called when the stream starts."""
        raise NotImplementedError

    def _task_workers(self) -> Sequence:
        """Return the workers that receive the tasks of a stream."""
        raise NotImplementedError

    def _send_task(self, worker, kind: str, payload: dict) -> None:
        """Send one task to a worker."""
        raise NotImplementedError

    def _receive_result(self, kind: str) -> tuple[Any, int, Any]:
        """Wait for the next task result: ``(worker, task_id, result)``; raise on a task failure."""
        raise NotImplementedError

    def _stop_stream(self, exc: BaseException, kind: str) -> None:
        """Deal with what a stream stopped by ``exc`` left on the workers; ``exc`` is raised again after it."""
        raise NotImplementedError

    def _end_stream(self) -> None:
        """Release what a stream that received every result still holds."""

    # Protocol shared by the runtimes.

    def broadcast_spec(self, kind: str, spec: dict) -> int:
        """Send a spec to every worker and wait until all have built its model (see ``TaskRuntime``)."""
        if self._stream is not None:
            raise RuntimeError("broadcast_spec while a task stream is open")
        self._prepare_broadcast()
        key = spec["specId"]
        if key in self._spec_ids:
            return self._spec_ids[key]
        spec_id = len(self._spec_ids)
        self._broadcast(kind, spec_id, spec)
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

        Only one iterator can be open at a time, also among iterators created before either started, and no
        spec can be broadcast while it is: exhaust or close it first (see ``TaskRuntime.stream_tasks``).
        """
        if self._stream is not None:
            raise RuntimeError("stream_tasks while another task stream is open")
        self._check_usable()

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

    def _stream_tasks(
        self, kind: str, payloads: Sequence[dict], *, progress: str | None, unit: str, sizes: Sequence[int] | None
    ) -> Iterator[Any]:
        from lunarops import console

        self._begin_stream()
        payloads = [dict(p, taskId=i) for i, p in enumerate(payloads)]
        counts = [1] * len(payloads) if sizes is None else [int(size) for size in sizes]
        workers = self._task_workers()
        results: dict[int, Any] = {}
        next_task = 0

        def send_next(worker) -> None:
            nonlocal next_task
            if next_task < len(payloads):
                self._send_task(worker, kind, payloads[next_task])
                next_task += 1

        with console.progress(total=sum(counts), desc=progress or "", unit=unit, disable=progress is None) as bar:
            try:
                for _ in range(QUEUED_TASKS_PER_WORKER):
                    for worker in workers:
                        send_next(worker)
                for wanted in range(len(payloads)):
                    while wanted not in results:
                        worker, task_id, result = self._receive_result(kind)
                        results[task_id] = result
                        bar.update(counts[task_id])
                        send_next(worker)
                    yield results.pop(wanted)
            except BaseException as exc:
                next_task = len(payloads)
                self._stop_stream(exc, kind)
                raise
        self._end_stream()


__all__ = [
    "KILL_TIMEOUT_S",
    "NATIVE_THREAD_VARIABLES",
    "QUEUED_TASKS_PER_WORKER",
    "TaskRuntime",
    "TaskRuntimeBase",
    "Terminated",
    "configure_native_threads",
    "describe_exit",
    "is_interrupt",
    "raise_terminated",
    "starting_children",
]
