"""Backward arc of a two-sided propagation in a process of its own.

The backward arc of a two-sided propagation depends on the forward arc only
through the first forward nodes it mirrors (see
:meth:`~lunarops.classes.dynamics.integrators.AdamsBashforthMoultonIntegrator.integrate`).
:class:`BackwardArcProcess` propagates it in a spawned process while this
process continues the forward arc. Threads would not help: the fused
right-hand-side kernel keeps per-evaluation state, and most of a step holds
the GIL.

The compiled dynamics do not pickle, so the arc process builds its own
problem from a picklable factory. It is started before this process builds
its problem, so that the two builds overlap. Once the mirrored forward nodes
are accepted, this process sends them; the arc process then runs exactly the
sequence of a sequential backward arc, node diagnostic included, and sends
back its nodes, the history counters and the diagnostic. The propagated
nodes, the diagnostics and the history counters are therefore bitwise
identical to a sequential propagation. A digest of the epoch inputs, the body
set and the integrator settings checks that both processes built the same
problem.

The arc process builds the epoch data of its arc itself, or, with
``workers`` > 0, on worker processes of its own (a
:class:`~lunarops.parallel.process.ProcessRuntime`).

Messages on the one duplex pipe (pickled tuples): this process sends
``("seed", initial_epoch, backward_duration_s, mirrored, block, digest,
diagnostic)``; the arc process answers ``("ready", digest, build_s,
worker_pids)`` once its problem is built, ``("progress", nodes)`` after each
epoch batch, and ``("done", ArcResult)`` or ``("error", exception,
traceback)``.

The arc process ignores SIGINT (Ctrl-C in a terminal signals the whole
process group; this process decides), turns SIGTERM into ``SystemExit`` so
that its own workers are stopped, and exits when this process closes the pipe
or dies.

Attributes
----------
STALL_TIMEOUT_S : float
    Time :meth:`BackwardArcProcess.finish` waits for any message of the arc
    process before it stops the process and raises.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import pickle
import signal
import sys
import time
import traceback
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from lunarops.classes.time import Epoch

    from .integrators import AdamsBashforthMoultonIntegrator, IntegratorSettings
    from .lunar_dynamics import LunarDynamics

STALL_TIMEOUT_S = 900.0

_JOIN_TIMEOUT_S = 10.0
"""Time a finished arc process has to exit before it is terminated."""

_KILL_TIMEOUT_S = 5.0
"""Time a terminated arc process has to exit before it is killed."""


@dataclass(frozen=True)
class ArcSetup:
    """Problem built by the factory of a :class:`BackwardArcProcess`, in the arc process.

    Attributes
    ----------
    system : LunarDynamics
        Dynamics, built as in the parent process.
    integrator : AdamsBashforthMoultonIntegrator
        Integrator, with the settings of the parent process.
    epoch_source_spec : dict or None, default None
        ``{"ephemerides": <class config>, "workingDir": <path>}`` from which worker processes open the
        ephemeris; required when the arc uses workers.
    close : Callable[[], None] or None, default None
        Called when the arc process ends, to release the resources of the problem.
    """

    system: LunarDynamics
    integrator: AdamsBashforthMoultonIntegrator
    epoch_source_spec: dict | None = None
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class ArcResult:
    """Propagated backward arc, as sent by the arc process.

    Attributes
    ----------
    nodes : tuple of numpy.ndarray
        Accepted backward nodes after the mirrored ones, in integration order, as
        :meth:`~lunarops.classes.dynamics.history.LunarNodeBuffer.block` returns them.
    extrapolated_translation_queries : int
        Final counter of the backward-arc history.
    max_translation_extrapolation_s : float
        Final counter of the backward-arc history, in s.
    diagnostic : ArcDiagnostic or None
        Node diagnostic of the backward arc, after its last call.
    """

    nodes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    extrapolated_translation_queries: int
    max_translation_extrapolation_s: float
    diagnostic: ArcDiagnostic | None

    @property
    def count(self) -> int:
        """int: Number of backward nodes."""
        return len(self.nodes[0])


@runtime_checkable
class ArcDiagnostic(Protocol):
    """Node diagnostic that can run the backward arc in the arc process.

    It is called like a ``NodeDiagnosticCallback``. For a parallel backward arc the integrator sends
    :meth:`for_backward_arc` to the arc process, which binds it to its own dynamics and calls it at each
    backward node; the parent then merges the returned diagnostic with :meth:`merge_backward`. The object
    sent must pickle without the dynamics.
    """

    def __call__(self, epoch: Epoch, state: np.ndarray, batch, index: int, history) -> None:
        """Record one accepted node."""
        ...

    def for_backward_arc(self) -> ArcDiagnostic:
        """Return an empty, unbound diagnostic for the backward arc."""
        ...

    def bind(self, system: LunarDynamics) -> None:
        """Use ``system`` to evaluate the dynamics."""
        ...

    def merge_backward(self, other: ArcDiagnostic) -> None:
        """Append what ``other`` recorded on the backward arc, as if this diagnostic had been called."""
        ...


def system_digest(system: LunarDynamics, settings: IntegratorSettings) -> str:
    """Return a digest of what makes two builds of a propagation problem integrate identically.

    Parameters
    ----------
    system : LunarDynamics
        Dynamics.
    settings : IntegratorSettings
        Integrator settings.

    Returns
    -------
    str
        SHA-256 of the epoch inputs, the body names, the settings and the largest advanced delay.
    """
    data = (system.epoch_inputs, system.body_names, settings, system.advanced_delay_s)
    return hashlib.sha256(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


class _RemoteTraceback(Exception):
    """Formatted traceback of an exception raised in the arc process."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text

    def __str__(self) -> str:
        return self.text


def _portable(error: BaseException) -> BaseException:
    """Return ``error`` if it survives pickling, else a RuntimeError naming it."""
    try:
        pickle.loads(pickle.dumps(error))
    except Exception:  # noqa: BLE001 - any pickling failure: report the error by name instead
        return RuntimeError(f"{type(error).__name__}: {error}")
    return error


def _describe_exit(exitcode: int | None) -> str:
    if exitcode is None:
        return "closed its pipe"
    if exitcode < 0:
        try:
            return f"was killed by {signal.Signals(-exitcode).name}"
        except ValueError:
            return f"was killed by signal {-exitcode}"
    return f"exited with code {exitcode}"


class BackwardArcProcess:
    """Backward arc of a two-sided propagation, propagated in a spawned process.

    Pass it to :meth:`~lunarops.classes.dynamics.integrators.AdamsBashforthMoultonIntegrator.integrate` as
    ``backward_arc``, which calls :meth:`seed`, :meth:`poll` and :meth:`finish`. Use it as a context manager:
    leaving the context (also on an error or an interrupt) stops the arc process and its workers.

    Parameters
    ----------
    factory : Callable[[], ArcSetup]
        Builds the problem in the arc process: the dynamics and integrator of the propagation. It must pickle
        (a module-level function, or a ``functools.partial`` of one), since the process is spawned.
    workers : int, default 0
        Worker processes of the arc process for its epoch data; 0 builds them in the arc process.
    initializer : Callable[[], None] or None, default None
        Module-level function run in the arc process before ``factory``, and in each of its workers, for
        example to register model classes.

    Notes
    -----
    The ``spawn`` start method imports the ``__main__`` module again in the arc process: a script that
    propagates a parallel backward arc must guard its entry point with ``if __name__ == "__main__":``.
    """

    def __init__(
        self,
        factory: Callable[[], ArcSetup],
        *,
        workers: int = 0,
        initializer: Callable[[], None] | None = None,
    ) -> None:
        if workers < 0:
            raise ValueError(f"workers must be non-negative, got {workers}")
        self._factory = factory
        self._workers = int(workers)
        self._initializer = initializer
        self._context = multiprocessing.get_context("spawn")
        self._process: Any = None
        self._pid: int | None = None
        self._conn: Connection | None = None
        self._ready: tuple[str, float, list[int]] | None = None
        self._result: ArcResult | None = None
        self._seeded = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def pid(self) -> int | None:
        """Process ID of the arc process once started (also after :meth:`close`), else None."""
        return self._pid

    @property
    def worker_pids(self) -> list[int]:
        """Process IDs of the workers of the arc process, once it has reported ready."""
        return [] if self._ready is None else list(self._ready[2])

    def start(self) -> None:
        """Start the arc process, which builds its problem and waits for :meth:`seed`.

        Raises
        ------
        RuntimeError
            If the process was already started, or if this process is daemonic (and may not start
            processes).
        """
        if self._process is not None:
            raise RuntimeError("The backward-arc process was already started")
        if multiprocessing.current_process().daemon:
            raise RuntimeError("A daemonic process cannot start the backward-arc process")
        conn, child = self._context.Pipe(duplex=True)
        # Not daemonic: the arc process starts worker processes of its own.
        self._process = self._context.Process(
            target=_arc_main,
            args=(child, self._factory, self._workers, self._initializer),
            name="lunarops-backward-arc",
            daemon=False,
        )
        self._conn = conn
        try:
            self._process.start()
            self._pid = self._process.pid
        finally:
            # Only the arc process holds its end: when it exits, reading this end meets the end of the pipe.
            child.close()

    def seed(
        self,
        initial_epoch: Epoch,
        backward_duration_s: float,
        mirrored: int,
        block,
        digest: str,
        diagnostic: ArcDiagnostic | None = None,
    ) -> None:
        """Send the mirrored forward nodes; the arc process then propagates. Starts the process if needed.

        Parameters
        ----------
        initial_epoch : Epoch
            TDB epoch of offset 0.
        backward_duration_s : float
            Positive length of the backward arc, in s.
        mirrored : int
            Number of mirrored forward nodes.
        block : tuple of numpy.ndarray
            The first ``mirrored`` forward nodes, from the initial epoch on, as
            :meth:`~lunarops.classes.dynamics.history.LunarNodeBuffer.block` returns them.
        digest : str
            :func:`system_digest` of the dynamics and settings of this process.
        diagnostic : ArcDiagnostic or None, optional
            Unbound node diagnostic for the backward arc.
        """
        if self._process is None:
            self.start()
        if self._seeded:
            raise RuntimeError("The backward arc was already seeded")
        self._seeded = True
        message = ("seed", initial_epoch, float(backward_duration_s), int(mirrored), block, digest, diagnostic)
        data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        assert self._conn is not None
        try:
            self._conn.send_bytes(data)
        except OSError:
            # The process has gone; its last message, if any, says why.
            self._raise_exit(None)

    def poll(self, progress: Callable[[int], None] | None = None) -> None:
        """Handle the messages the arc process has sent so far, without waiting.

        Parameters
        ----------
        progress : Callable[[int], None] or None, optional
            Called with the number of backward nodes accepted since the last report.

        Raises
        ------
        RuntimeError
            If the arc process has exited before finishing.
        Exception
            The exception raised in the arc process, chained to its remote traceback.
        """
        if self._process is None or self._result is not None:
            return
        assert self._conn is not None
        # Only the arc process holds the other end, so its exit makes this end readable: one cheap check per
        # forward node is enough.
        while self._result is None and self._conn.poll():
            self._handle(self._receive(progress), progress)

    def finish(self, progress: Callable[[int], None] | None = None) -> ArcResult:
        """Wait for the backward arc and return it.

        Waits as long as the arc process sends a message at least every :data:`STALL_TIMEOUT_S`.

        Parameters
        ----------
        progress : Callable[[int], None] or None, optional
            Called with the number of backward nodes accepted since the last report.

        Returns
        -------
        ArcResult
            Backward nodes, history counters and diagnostic.

        Raises
        ------
        RuntimeError
            If the arc was never seeded, the arc process exited before finishing, or it sent nothing for
            :data:`STALL_TIMEOUT_S` (it is then stopped).
        Exception
            The exception raised in the arc process, chained to its remote traceback.
        """
        if not self._seeded:
            raise RuntimeError("The backward arc was never seeded")
        assert self._conn is not None
        sentinel = self._process.sentinel
        timeout = STALL_TIMEOUT_S
        while self._result is None:
            ready = wait([self._conn, sentinel], timeout=timeout)
            if self._conn in ready:
                self._handle(self._receive(progress), progress)
            elif sentinel in ready:
                self._raise_exit(progress)
            else:
                self._stop()
                raise RuntimeError(f"The backward-arc process sent nothing for {timeout:g} s and was stopped")
        return self._result

    def close(self) -> None:
        """Stop the arc process if it is still running, and release the process and the pipe."""
        process, self._process = self._process, None
        if process is not None and process.pid is not None:
            if self._result is not None:
                # Finished: it is stopping its own workers.
                process.join(_JOIN_TIMEOUT_S)
            self._stop(process)
            process.close()
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    def _stop(self, process=None) -> None:
        process = self._process if process is None else process
        if process.exitcode is None:
            process.terminate()
            process.join(_KILL_TIMEOUT_S)
            if process.exitcode is None:
                process.kill()
        process.join()

    def _receive(self, progress) -> tuple | None:
        """Return the next message, or None if the end of the pipe was reached after the result."""
        assert self._conn is not None
        try:
            data = self._conn.recv_bytes()
        except EOFError, OSError:
            self._raise_exit(progress)
            return None
        return pickle.loads(data)

    def _handle(self, message: tuple | None, progress) -> None:
        if message is None:
            return
        kind = message[0]
        if kind == "progress":
            if progress is not None:
                progress(message[1])
        elif kind == "ready":
            self._ready = message[1:]
        elif kind == "done":
            self._result = message[1]
        elif kind == "error":
            _, error, text = message
            error.__cause__ = _RemoteTraceback(text)
            raise error
        else:
            raise RuntimeError(f"Unexpected {kind!r} message from the backward-arc process")

    def _raise_exit(self, progress):
        """Handle the messages still in the pipe (an error may be waiting), then report the exit."""
        assert self._conn is not None
        while self._result is None:
            try:
                if not self._conn.poll():
                    break
                data = self._conn.recv_bytes()
            except EOFError, OSError:
                break
            self._handle(pickle.loads(data), progress)
        if self._result is not None:
            return
        self._process.join(_KILL_TIMEOUT_S)
        raise RuntimeError(
            f"The backward-arc process {_describe_exit(self._process.exitcode)} before finishing"
        ) from None


class _ParentGone(Exception):
    """The parent process closed the pipe or died."""


def _send(conn: Connection, message) -> None:
    try:
        conn.send_bytes(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))
    except OSError:
        raise _ParentGone from None


def _terminated(signum, frame):
    raise SystemExit(128 + signum)


class _ProgressReporter:
    """``after_node`` hook of the arc process: reports the accepted nodes after each epoch batch.

    Also checks once per batch whether the parent is still there: it sends nothing after the seed, so a
    readable pipe means that it has closed it.
    """

    def __init__(self, conn: Connection, batch_size: int, last: int) -> None:
        self._conn = conn
        self._batch_size = batch_size
        self._last = last
        self._pending = 0

    def __call__(self, nodes, index: int) -> None:
        self._pending += 1
        if index % self._batch_size and index != self._last:
            return
        if self._conn.poll():
            raise _ParentGone
        _send(self._conn, ("progress", self._pending))
        self._pending = 0


def _propagate(conn: Connection, factory, workers: int, initializer) -> ArcResult:
    from .integrators import _EPOCH_BATCH_SIZE
    from .trajectory import regular_times

    with ExitStack() as stack:
        started = time.perf_counter()
        if initializer is not None:
            initializer()
        setup = factory()
        if setup.close is not None:
            stack.callback(setup.close)
        system, integrator = setup.system, setup.integrator
        source, pids = None, []
        if workers:
            from lunarops.parallel.epoch_stream import RuntimeEpochSource
            from lunarops.parallel.process import ProcessRuntime

            if setup.epoch_source_spec is None:
                raise ValueError("A backward arc with workers needs the epoch_source_spec of its ArcSetup")
            runtime = ProcessRuntime(workers, initializer=initializer)
            # Any failure, SIGTERM included, stops the workers at once.
            stack.push(lambda exc_type, exc, tb: runtime.shutdown(interrupted=exc is not None))
            pids = runtime.pids
            source = RuntimeEpochSource(
                runtime,
                system.epoch_inputs,
                setup.epoch_source_spec["ephemerides"],
                setup.epoch_source_spec["workingDir"],
                "backward-arc",
            )
        digest = system_digest(system, integrator.settings)
        _send(conn, ("ready", digest, time.perf_counter() - started, pids))
        try:
            message = pickle.loads(conn.recv_bytes())
        except EOFError, OSError:
            raise _ParentGone from None
        _, initial_epoch, backward_duration_s, mirrored, block, parent_digest, diagnostic = message
        if parent_digest != digest:
            raise RuntimeError(
                "The backward-arc process built a different problem from the parent process (digest "
                f"{digest[:12]} against {parent_digest[:12]})"
            )
        if integrator._mirrored_node_count(system) != mirrored:
            raise RuntimeError("The backward-arc process mirrors a different number of forward nodes")
        times = regular_times(-backward_duration_s, integrator.settings.step_s)
        nodes, history = integrator._backward_start(system, initial_epoch, block, times)
        if diagnostic is not None:
            diagnostic.bind(system)
        integrator._main_grid(
            system,
            initial_epoch,
            times,
            nodes,
            history,
            prefetch=None,
            epoch_source=source,
            node_diagnostic=diagnostic,
            dop853_steps=0,
            after_node=_ProgressReporter(conn, _EPOCH_BATCH_SIZE, len(times) - 1),
        )
        return ArcResult(
            nodes.block(mirrored, nodes.size),
            history.extrapolated_translation_queries,
            history.max_translation_extrapolation_s,
            diagnostic,
        )


def _arc_main(conn: Connection, factory, workers: int, initializer) -> None:
    """Build the problem, wait for the seed, propagate the backward arc and send the result."""
    # The parent decides on interruption and stops this process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _terminated)
    try:
        try:
            # The workers are stopped before the result is sent: "done" means this process is exiting.
            message = ("done", _propagate(conn, factory, workers, initializer))
            data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        except _ParentGone:
            return
        except Exception as error:  # noqa: BLE001 - every failure must reach the parent
            data = pickle.dumps(("error", _portable(error), traceback.format_exc()), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            conn.send_bytes(data)
        except OSError:
            pass  # The parent has gone.
    finally:
        conn.close()
        sys.stdout.flush()


__all__ = ["STALL_TIMEOUT_S", "ArcDiagnostic", "ArcResult", "ArcSetup", "BackwardArcProcess", "system_digest"]
