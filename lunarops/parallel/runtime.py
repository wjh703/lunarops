"""Interface shared by the task runtimes, and the process set-up of a parallel run.

A task runtime distributes model evaluations over worker processes: ``MpiRuntime`` over MPI ranks (``--mpi``)
and ``ProcessRuntime`` over local processes that it starts itself (``--workers N``). Programs see only the
``TaskRuntime`` interface: they broadcast a spec, from which every worker builds its model once, and then
stream small tasks that refer to the spec by ID. Both runtimes run the same task handlers
(:mod:`lunarops.parallel.tasks`), so a run gives the same results with either runtime and serially.

This module imports nothing heavy: ``configure_native_threads`` must run before NumPy is loaded.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

NATIVE_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
"""Environment variables that set the thread count of the native numerical libraries."""

_ORDERLY = (Exception, GeneratorExit, SystemExit)


def configure_native_threads() -> None:
    """Limit the native numerical libraries to one thread per process.

    Every process of a parallel run is one busy CPU; BLAS or OpenMP thread pools on top of it would
    oversubscribe the CPUs. The variables are read when the libraries load, so this must run before NumPy is
    imported; processes started afterwards inherit them.
    """
    for name in NATIVE_THREAD_VARIABLES:
        os.environ[name] = "1"


def is_interrupt(exc: BaseException) -> bool:
    """Return whether an exception interrupts a parallel run rather than ending it in an orderly way.

    Ordinary exceptions (``Exception``, which includes worker task failures), ``GeneratorExit`` (a task
    stream closed early) and ``SystemExit`` are orderly; any other ``BaseException``, ``KeyboardInterrupt``
    in practice, is an interrupt, on which the runtime stops its workers without waiting for their tasks.

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


__all__ = ["NATIVE_THREAD_VARIABLES", "TaskRuntime", "configure_native_threads", "is_interrupt"]
