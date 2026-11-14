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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

NATIVE_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
"""Environment variables that set the thread count of the native numerical libraries."""

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


__all__ = [
    "KILL_TIMEOUT_S",
    "NATIVE_THREAD_VARIABLES",
    "TaskRuntime",
    "Terminated",
    "configure_native_threads",
    "describe_exit",
    "is_interrupt",
    "raise_terminated",
    "starting_children",
]
