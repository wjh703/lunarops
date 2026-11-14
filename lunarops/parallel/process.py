"""Master-worker execution on local worker processes.

``ProcessRuntime`` starts N worker processes on this machine (``lunarops run --workers N``) and talks to each
through one duplex pipe. It offers the interface of ``MpiRuntime`` (``TaskRuntime``): the master broadcasts a
spec, from which every worker builds its model once, then streams small tasks that refer to the spec by ID,
and receives the results in task order. The workers keep their models in a ``WorkerState`` and run the task
handlers of :mod:`lunarops.parallel.tasks`, exactly as MPI workers do, so that serial, MPI and process runs
give the same results.

Messages (pickled tuples) from the master are ``("spec", kind, spec_id, spec)``, ``("task", kind, payload)``
and ``("stop",)``; the workers answer ``("ready", pid)`` once started, ``("spec_done", spec_id, seconds,
error)``, ``("result", task_id, result)`` and ``("error", task_id, traceback)``, or ``("fatal", traceback)``
before they give up on a message they cannot handle.

Errors and interrupts follow one rule. A stream stopped by an ordinary exception (a worker task failure
included) or closed early (``GeneratorExit``) receives the results the workers still owe it, and the runtime
stays usable. An interrupt (``is_interrupt``: ``KeyboardInterrupt`` or ``Terminated``), any exception raised
while a message is partly sent or received (the pipe is then out of step), a failure inside a spec broadcast,
a worker that dies, or workers that do not return what they owe within ``JOIN_TIMEOUT_S``, stops every worker
at once: the runtime is then broken and refuses further work. Workers start with SIGINT ignored
(``starting_children``) and keep ignoring it, so that Ctrl-C in a terminal, which signals the whole process
group, leaves the decision to the master; they exit when the master closes their pipe, also when it dies
without a chance to stop them.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing.connection import Connection, wait
from typing import Any, Self

from lunarops import console
from lunarops.parallel.runtime import KILL_TIMEOUT_S, describe_exit, is_interrupt, starting_children
from lunarops.parallel.tasks import SPEC_LABELS, TASK_HANDLERS, WorkerState

_STOP = pickle.dumps(("stop",), protocol=pickle.HIGHEST_PROTOCOL)

JOIN_TIMEOUT_S = 10.0
"""Time the workers have to exit after a stop message before they are terminated."""

_DRAIN_TIMEOUT_S = JOIN_TIMEOUT_S
"""Time the workers have to return the results they owe a stopped stream before they are stopped."""

_QUEUED_TASKS = 2
"""Number of tasks a worker holds at a time: one in work and one queued."""


def _dumps(message) -> bytes:
    return pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)


def _read_messages(conn: Connection, inbox: queue.SimpleQueue) -> None:
    # Reads every message as soon as it arrives, so that the master never waits to send a task while this
    # worker sends a large result. The master closes its end only after a stop message has been answered by
    # an exit, or once it has terminated the worker or died: the end of the pipe ends the worker at once,
    # also in the middle of a task.
    while True:
        try:
            data = conn.recv_bytes()
        except EOFError, OSError:
            os._exit(0)
        try:
            inbox.put(pickle.loads(data))
        except Exception:  # noqa: BLE001 -- reported by the worker loop
            inbox.put(("undecodable", traceback.format_exc()))


def _reply(conn: Connection, data: bytes) -> None:
    """Send a message to the master; exit at once if the master has gone (its pipe is broken)."""
    try:
        conn.send_bytes(data)
    except OSError:
        os._exit(0)


def _serve(message: tuple, state: WorkerState) -> bytes:
    tag = message[0]
    if tag == "task":
        _, kind, payload = message
        task_id = payload["taskId"]
        try:
            return _dumps(("result", task_id, TASK_HANDLERS[kind](payload, state)))
        except Exception:  # noqa: BLE001 -- forwarded to the master with its traceback
            return _dumps(("error", task_id, traceback.format_exc()))
    if tag == "spec":
        _, kind, spec_id, spec = message
        started = time.perf_counter()
        error = ""
        try:
            state.initialize(kind, spec_id, spec)
        except Exception:  # noqa: BLE001 -- forwarded to the master with its traceback
            error = traceback.format_exc()
        return _dumps(("spec_done", spec_id, time.perf_counter() - started, error))
    if tag == "undecodable":
        raise RuntimeError(f"undecodable message from the master:\n{message[1]}")
    raise RuntimeError(f"unknown message {tag!r} from the master")


def _worker_main(conn: Connection, initializer: Callable[[], None] | None) -> None:
    """Serve spec and task messages until a stop message arrives or the master closes the pipe.

    Parameters
    ----------
    conn : Connection
        Worker end of the duplex pipe to the master.
    initializer : callable or None
        Function run before the worker reports that it is ready.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if initializer is not None:
            initializer()
    except Exception:  # noqa: BLE001 -- reported by the master
        _reply(conn, _dumps(("init_error", traceback.format_exc())))
        return
    inbox: queue.SimpleQueue = queue.SimpleQueue()
    threading.Thread(target=_read_messages, args=(conn, inbox), name="lunarops-worker-reader", daemon=True).start()
    _reply(conn, _dumps(("ready", os.getpid())))
    state = WorkerState()
    try:
        while (message := inbox.get())[0] != "stop":
            try:
                reply = _serve(message, state)
            except Exception:  # noqa: BLE001 -- a malformed message: report it and give up
                _reply(conn, _dumps(("fatal", traceback.format_exc())))
                raise SystemExit(1) from None
            _reply(conn, reply)
    finally:
        state.close()


class _Worker:
    """Master-side record of one worker process."""

    __slots__ = ("conn", "number", "outstanding", "pid", "process", "released")

    def __init__(self, number: int, process, conn: Connection) -> None:
        self.number = number
        self.process = process
        self.conn = conn
        self.outstanding = 0
        self.pid: int | None = None
        # Once released, the process object is closed and may no longer be used.
        self.released = False

    def __str__(self) -> str:
        return f"worker process {self.number} (pid {self.pid})"


class ProcessRuntime:
    """Master-worker task execution on local worker processes.

    The workers are started with the ``spawn`` method, so that they start from a fresh interpreter whatever
    this process has loaded; each runs ``initializer`` (if given) and then serves the master. Each worker
    holds up to two tasks at a time and receives the next one when it returns a result.

    Parameters
    ----------
    workers : int
        Number of worker processes, at least 1.
    initializer : callable, optional
        Function called without arguments in every worker before it serves tasks, for example to register
        model classes. It must be importable by name (a module-level function), since the workers are
        spawned.
    start_timeout_s : float, optional
        Time the workers have to start and run ``initializer``. Default 120 s.

    Attributes
    ----------
    name : str
        Runtime name in run metadata, ``process``.

    Raises
    ------
    ValueError
        If ``workers`` is less than 1.
    RuntimeError
        If a worker fails to start or to run ``initializer``, or does not start in time; the workers started
        are stopped.

    Notes
    -----
    Programs started through the ``spawn`` method import the ``__main__`` module of the master again: a
    script that creates a ``ProcessRuntime`` must guard its entry point with ``if __name__ == "__main__":``.
    """

    name = "process"

    def __init__(
        self,
        workers: int,
        *,
        initializer: Callable[[], None] | None = None,
        start_timeout_s: float = 120.0,
    ) -> None:
        count = int(workers)
        if count < 1:
            raise ValueError(f"ProcessRuntime needs at least one worker, got {workers}")
        self._context = multiprocessing.get_context("spawn")
        self._initializer = initializer
        self._workers: list[_Worker] = []
        self._active = count
        self._spec_ids: dict[str, int] = {}
        self._stream: Generator | None = None
        self._broken: str | None = None
        self._closed = False
        # Worker whose pipe a message is being sent on or received from: an exception raised meanwhile leaves
        # the pipe out of step, and breaks the runtime instead of draining it.
        self._in_flight: _Worker | None = None
        try:
            self._start(count, initializer, start_timeout_s)
        except BaseException:
            self._closed = True
            self._terminate()
            self._release()
            raise

    def _start(self, count: int, initializer, timeout_s: float) -> None:
        for number in range(1, count + 1):
            conn, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_worker_main, args=(child, initializer), name=f"lunarops-worker-{number}", daemon=True
            )
            worker = _Worker(number, process, conn)
            self._workers.append(worker)
            try:
                with starting_children():
                    process.start()
                    worker.pid = process.pid
            finally:
                # Only the worker holds its end: when it exits, reading this end meets the end of the pipe.
                child.close()
        deadline = time.monotonic() + timeout_s
        waiting = list(self._workers)
        while waiting:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{len(waiting)} of {count} worker processes did not start within {timeout_s:g} s")
            ready = set(wait([w.conn for w in waiting] + [w.process.sentinel for w in waiting], timeout=remaining))
            for worker in [w for w in waiting if w.conn in ready or w.process.sentinel in ready]:
                message = self._recv(worker, "start-up") if worker.conn.poll() else None
                if message is None:
                    raise self._dead_worker(worker, "start-up")
                if message[0] == "init_error":
                    raise RuntimeError(f"{worker} failed to run its initializer:\n{message[1]}")
                if message[0] != "ready":
                    raise RuntimeError(f"{worker} sent {message[0]!r} during start-up")
                waiting.remove(worker)

    @property
    def workers(self) -> int:
        """Number of worker processes."""
        return len(self._workers)

    @property
    def initializer(self) -> Callable[[], None] | None:
        """Initializer run in every worker, for processes that start workers of their own."""
        return self._initializer

    @property
    def pids(self) -> list[int]:
        """Process IDs of the workers (also after ``shutdown``)."""
        return [worker.pid for worker in self._workers if worker.pid is not None]

    def describe(self) -> str:
        """Return the execution mode shown in the run header.

        Returns
        -------
        str
            For example ``4 worker processes``.
        """
        return "1 worker process" if self.workers == 1 else f"{self.workers} worker processes"

    @contextmanager
    def limit(self, workers: int) -> Iterator[None]:
        """Distribute the tasks of the streams started inside the block over the first ``workers`` workers.

        Specs are still broadcast to every worker.

        Parameters
        ----------
        workers : int
            Number of workers that receive tasks, from 1 to ``self.workers``.

        Yields
        ------
        None
            Inside the block, new streams use only the selected workers.

        Raises
        ------
        ValueError
            If ``workers`` is out of range.
        RuntimeError
            If a task stream is open.
        """
        if not 1 <= workers <= self.workers:
            raise ValueError(f"limit must be between 1 and {self.workers} workers, got {workers}")
        if self._stream is not None:
            raise RuntimeError("limit while a task stream is open")
        previous, self._active = self._active, workers
        try:
            yield
        finally:
            self._active = previous

    def broadcast_spec(self, kind: str, spec: dict) -> int:
        """Send a spec to every worker and wait until all have built its model.

        A spec whose ``specId`` was already broadcast is not sent again. A worker that fails to build the
        model reports its traceback, and the runtime stays usable; any other failure during the broadcast
        (an interrupt, a worker that dies) stops the workers.

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
            If a task stream is open, the runtime is broken or shut down, or a worker fails to build the
            model.
        """
        if self._stream is not None:
            raise RuntimeError("broadcast_spec while a task stream is open")
        self._check_usable()
        key = spec["specId"]
        if key in self._spec_ids:
            return self._spec_ids[key]
        spec_id = len(self._spec_ids)
        label = SPEC_LABELS.get(kind, kind)
        during = f"the {label} broadcast"
        data = _dumps(("spec", kind, spec_id, spec))
        timings: list[float] = []
        errors: list[str] = []
        try:
            for worker in self._workers:
                self._send(worker, data, during)
            while any(worker.outstanding for worker in self._workers):
                worker, message = self._receive(during)
                if message[0] != "spec_done" or message[1] != spec_id:
                    raise self._protocol_error(worker, message, during)
                _, _, seconds, error = message
                timings.append(seconds)
                if error:
                    errors.append(f"{worker} failed initializing {kind}:\n{error}")
        except BaseException as exc:
            self._break(f"{type(exc).__name__} during {during}")
            raise
        if errors:
            raise RuntimeError(errors[0])
        console.info(f"{label} ready on {self.describe()} ({max(timings):.2f} s)")
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
        first. An iterator stopped by an error or closed early receives the results still owed to it before
        it ends (the workers have ``JOIN_TIMEOUT_S`` to return them, or are stopped); one stopped by an
        interrupt (``is_interrupt``), or by any exception in the middle of a message, stops the workers
        instead.

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
            If another task stream is open (when called, or when the iterator starts), the runtime is broken
            or shut down, or, while iterating, if a worker reports a task failure or dies.
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

    def _stream_tasks(
        self, kind: str, payloads: Sequence[dict], *, progress: str | None, unit: str, sizes: Sequence[int] | None
    ) -> Iterator[Any]:
        self._check_usable()
        payloads = [dict(p, taskId=i) for i, p in enumerate(payloads)]
        counts = [1] * len(payloads) if sizes is None else [int(size) for size in sizes]
        workers = self._workers[: self._active]
        during = f"{kind} tasks"
        results: dict[int, Any] = {}
        next_task = 0

        def send_next(worker: _Worker) -> None:
            nonlocal next_task
            if next_task < len(payloads):
                self._send(worker, _dumps(("task", kind, payloads[next_task])), during)
                next_task += 1

        with console.progress(total=sum(counts), desc=progress or "", unit=unit, disable=progress is None) as bar:
            try:
                for _ in range(_QUEUED_TASKS):
                    for worker in workers:
                        send_next(worker)
                for wanted in range(len(payloads)):
                    while wanted not in results:
                        worker, message = self._receive(during)
                        if message[0] not in ("result", "error"):
                            raise self._protocol_error(worker, message, during)
                        _, task_id, value = message
                        if message[0] == "error":
                            raise RuntimeError(f"{worker} failed on {kind} task {task_id}:\n{value}")
                        results[task_id] = value
                        bar.update(counts[task_id])
                        send_next(worker)
                    yield results.pop(wanted)
            except BaseException as exc:
                next_task = len(payloads)
                if is_interrupt(exc) or self._in_flight is not None:
                    self._break(f"{type(exc).__name__} during {during}")
                else:
                    self._drain(during)
                raise

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

    def shutdown(self, *, interrupted: bool = False) -> None:
        """Stop the workers; later calls do nothing.

        After success or an ordinary exception, an open task stream is closed (it receives the results still
        owed to it, within ``JOIN_TIMEOUT_S``), every worker receives a stop message and has ``JOIN_TIMEOUT_S``
        to exit. With ``interrupted``, or once the runtime is broken, the workers are terminated at once.
        Workers still running at the end are terminated, and killed if they do not exit.

        Parameters
        ----------
        interrupted : bool, optional
            Whether the run was interrupted.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if interrupted:
                self._break("interrupted")
            if self._stream is not None:
                self._stream.close()
            if self._broken is None:
                for worker in self._workers:
                    try:
                        worker.conn.send_bytes(_STOP)
                    except OSError:
                        pass
                deadline = time.monotonic() + JOIN_TIMEOUT_S
                for worker in self._workers:
                    worker.process.join(max(0.0, deadline - time.monotonic()))
        finally:
            self._terminate()
            self._release()
            self._spec_ids.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(interrupted=exc is not None and is_interrupt(exc))

    def _check_usable(self) -> None:
        if self._closed:
            raise RuntimeError("the process runtime is shut down")
        if self._broken is not None:
            raise RuntimeError(f"the worker processes were stopped ({self._broken}); start a new ProcessRuntime")

    def _send(self, worker: _Worker, data: bytes, during: str) -> None:
        self._in_flight = worker
        try:
            worker.conn.send_bytes(data)
        except OSError:
            raise self._dead_worker(worker, during) from None
        worker.outstanding += 1
        self._in_flight = None

    def _recv(self, worker: _Worker, during: str) -> tuple | None:
        """Return the next message of a worker, or None at the end of its pipe."""
        try:
            data = worker.conn.recv_bytes()
        except EOFError, OSError:
            return None
        try:
            return pickle.loads(data)
        except Exception as exc:
            self._break(f"undecodable message from {worker}")
            raise RuntimeError(f"undecodable message from {worker} during {during}") from exc

    def _receive(self, during: str) -> tuple[_Worker, tuple]:
        """Wait for the next reply of a worker that owes one, which it then owes no longer.

        A worker that dies on the way, or reports a fatal error, breaks the runtime.
        """
        reply = self._wait_reply(during, None)
        assert reply is not None
        return reply

    def _wait_reply(self, during: str, deadline: float | None) -> tuple[_Worker, tuple] | None:
        """``_receive``, or None if no reply arrives before ``deadline`` (``time.monotonic``)."""
        while True:
            busy = [worker.conn for worker in self._workers if worker.outstanding]
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready = set(wait(busy + [worker.process.sentinel for worker in self._workers], timeout))
            if not ready:
                return None
            for worker in self._workers:
                # A worker may have exited right after its last message: read the pipe before the sentinel.
                if worker.conn in ready or (worker.process.sentinel in ready and worker.conn.poll()):
                    self._in_flight = worker
                    message = self._recv(worker, during)
                    if message is None:
                        raise self._dead_worker(worker, during)
                    if message[0] == "fatal":
                        self._break(f"{worker} failed")
                        raise RuntimeError(f"{worker} failed during {during}:\n{message[1]}")
                    worker.outstanding -= 1
                    self._in_flight = None
                    return worker, message
            for worker in self._workers:
                if worker.process.sentinel in ready:
                    raise self._dead_worker(worker, during)

    def _drain(self, during: str) -> None:
        """Receive and discard the replies the workers still owe, after a stream stopped early.

        The workers have ``_DRAIN_TIMEOUT_S`` to send them; on expiry, or if an exception (a second Ctrl-C)
        arrives meanwhile, or if the pipes are out of step, the workers are stopped instead.
        """
        if self._broken is not None:
            return
        if self._in_flight is not None:
            self._break(f"stopped in the middle of a message during {during}")
            return
        deadline = time.monotonic() + _DRAIN_TIMEOUT_S
        try:
            while any(worker.outstanding for worker in self._workers):
                reply = self._wait_reply(during, deadline)
                if reply is None:
                    console.warning(
                        f"the worker processes did not return the results of the stopped {during} within "
                        f"{_DRAIN_TIMEOUT_S:g} s; they are stopped"
                    )
                    self._break(f"results of stopped {during} not returned within {_DRAIN_TIMEOUT_S:g} s")
                    return
                worker, message = reply
                if message[0] not in ("result", "error"):
                    raise self._protocol_error(worker, message, during)
        except BaseException as exc:
            self._break(f"{type(exc).__name__} while draining {during}")
            raise

    def _dead_worker(self, worker: _Worker, during: str) -> RuntimeError:
        worker.process.join(KILL_TIMEOUT_S)
        error = RuntimeError(f"{worker} {describe_exit(worker.process.exitcode)} during {during}")
        self._break(str(error))
        return error

    def _protocol_error(self, worker: _Worker, message: tuple, during: str) -> RuntimeError:
        error = RuntimeError(f"unexpected {message[0]!r} message from {worker} during {during}")
        self._break(str(error))
        return error

    def _break(self, reason: str) -> None:
        """Mark the runtime broken and stop every worker."""
        if self._broken is None:
            self._broken = reason
        for worker in self._workers:
            worker.outstanding = 0
        self._in_flight = None
        self._terminate()

    def _terminate(self) -> None:
        started = [w.process for w in self._workers if w.pid is not None and not w.released]
        for process in started:
            if process.exitcode is None:
                process.terminate()
        deadline = time.monotonic() + KILL_TIMEOUT_S
        for process in started:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in started:
            if process.exitcode is None:
                process.kill()
                process.join(1.0)

    def _release(self) -> None:
        """Close the pipes, and the process objects of the workers that have exited (their sentinels)."""
        for worker in self._workers:
            worker.conn.close()
            if worker.pid is not None and not worker.released and worker.process.exitcode is not None:
                worker.process.close()
                worker.released = True


__all__ = ["JOIN_TIMEOUT_S", "ProcessRuntime"]
