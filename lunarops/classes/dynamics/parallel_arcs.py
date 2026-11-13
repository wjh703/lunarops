"""Concurrent backward arc of a two-sided propagation.

The backward arc of a two-sided propagation depends on the forward arc only
through the first forward nodes it mirrors (see
:meth:`~lunarops.classes.dynamics.integrators.AdamsBashforthMoultonIntegrator.integrate`).
Once these are accepted, a forked child process propagates the backward arc
while this process continues the forward one. Threads would not help: the
fused right-hand-side kernel keeps per-evaluation state, and most of a step
holds the GIL.

The epoch data of the backward arc come from the producer a sequential run
would use. Built in process, they are built in the child. From an epoch
source (the MPI workers), this process interleaves the backward batches with
the forward ones in its own stream and relays them to the child, which never
uses MPI.

After each epoch batch the child sends its accepted nodes back, with the
history counters after each node. This process appends the nodes, one by one,
to a mirror of the backward node buffer and calls the node diagnostic with the
mirror's history, which then holds exactly the nodes and counters a
sequential run would show at that node. The propagated nodes, the diagnostics
and the history counters are bitwise identical to a sequential propagation.

Both directions are plain pipes, each with a single writer. This process
writes the relayed batches from the integrating thread and starts no thread
of its own, and a full pipe holds it back until the child catches up. The
child writes through a sender thread, so that it never waits for this
process. When either side dies, the other reads the end of the pipe or fails
to write, and reports it instead of waiting.
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import pickle
import queue
import signal
import sys
import threading
import traceback
import warnings
from collections import deque
from collections.abc import Iterator
from itertools import zip_longest
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Self

import numpy as np

from .history import DelayedHistory, LunarNodeBuffer

_PIPE_BYTES = 1 << 20
"""Requested pipe capacity, the default ``pipe-max-size`` for unprivileged processes.

A relayed batch of the full DE440 body set is about 2 MB, so sending one waits until the child has started
reading it; this keeps the relay in step with the child.
"""

_PR_SET_PDEATHSIG = 1


def parallel_arcs_available() -> bool:
    """Return whether the backward arc can run in a forked process here.

    Requires Linux (``fork`` of a process with loaded numerical libraries is only safe there, and the child uses
    ``prctl``), at least two usable CPUs (a rank bound to one core would share it with the child), a
    non-daemonic process (which may not start children).
    """
    if not sys.platform.startswith("linux") or multiprocessing.current_process().daemon:
        return False
    if len(os.sched_getaffinity(0)) < 2:
        return False
    return True


class _RemoteTraceback(Exception):
    """Formatted traceback of an exception raised in the backward-arc process."""

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


def _pipe(context) -> tuple[Connection, Connection]:
    reader, writer = context.Pipe(duplex=False)
    try:
        import fcntl

        fcntl.fcntl(writer.fileno(), getattr(fcntl, "F_SETPIPE_SZ", 1031), _PIPE_BYTES)
    except ImportError, OSError:
        pass  # A smaller pipe only means more hand-offs per message.
    return reader, writer


def _die_with_parent(parent_pid: int) -> None:
    """Have the kernel kill this process when the parent exits."""
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except OSError, AttributeError:
        pass
    if os.getppid() != parent_pid:
        os._exit(1)


class _Sender:
    """Writes the child's messages from a thread; messages are pickled by the caller's thread."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._queue: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="lunarops-backward-arc-sender", daemon=True)
        self._thread.start()

    def send(self, message) -> None:
        # Pickling here, inside the caller's error handling, means no message can be dropped unnoticed.
        self._queue.put(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._connection.close()

    def _run(self) -> None:
        while (data := self._queue.get()) is not None:
            try:
                self._connection.send_bytes(data)
            except OSError:
                return  # The parent has gone.


class _RelayedEpochSource:
    """Epoch source of the child: the batches relayed by the parent, in the order of the requests."""

    def __init__(self, stash: list, batches: Connection) -> None:
        self._stash = stash
        self._batches = batches

    def stream(self, initial_epoch, offset_batches):
        for index, offsets in enumerate(offset_batches):
            if index < len(self._stash):
                relayed, batch = self._stash[index]
            else:
                relayed, batch = pickle.loads(self._batches.recv_bytes())
            if relayed != list(offsets):
                raise RuntimeError("A relayed epoch batch does not match the backward-arc request")
            yield batch


class _NodeReporter:
    """Node diagnostic of the child: sends the nodes accepted with each epoch batch to the parent."""

    def __init__(self, sender: _Sender, nodes, *, send_batches: bool) -> None:
        self._sender = sender
        self._nodes = nodes
        self._send_batches = send_batches
        self._batch = None
        self._start = nodes.size
        self._slots: list[int] = []
        self._counters: list[tuple[int, float]] = []

    def __call__(self, epoch, state, batch, slot, history) -> None:
        # Called after the node is appended: a new batch means the nodes of the previous one are complete.
        if batch is not self._batch:
            self.flush()
            self._batch = batch
        self._slots.append(slot)
        self._counters.append((history.extrapolated_translation_queries, history.max_translation_extrapolation_s))

    def flush(self) -> None:
        if not self._slots:
            return
        stop = self._start + len(self._slots)
        block = self._nodes.block(self._start, stop)
        batch = self._batch if self._send_batches else None
        self._sender.send(("nodes", block, self._slots, self._counters, batch))
        self._start, self._slots, self._counters = stop, [], []


class _ConcurrentBackwardArc:
    """Backward arc of a two-sided propagation, propagated in a forked process.

    Use as a context manager around the forward main grid: pass
    :attr:`forward_source` as its epoch source and :meth:`after_forward_node`
    as its ``after_node`` hook, then call :meth:`finish`. Leaving the context
    stops the child process if it is still running.

    Parameters
    ----------
    integrator : AdamsBashforthMoultonIntegrator
        Integrator propagating both arcs.
    system : LunarDynamics
        Dynamics being propagated.
    initial_epoch : Epoch
        TDB epoch of offset 0.
    backward_times : numpy.ndarray
        Main-grid offsets of the backward arc, from 0 outwards, in s.
    mirrored : int
        Number of forward nodes, from the initial epoch on, that seed the backward arc; at least 2.
    chunks : tuple of list
        Epoch-batch offsets of the forward and backward main grids, in request order.
    epoch_source : object or None
        Epoch source of the propagation; ``None`` builds the data in process.
    node_diagnostic : NodeDiagnosticCallback or None
        Node diagnostic of the propagation, called in this process.
    """

    def __init__(
        self,
        integrator,
        system,
        initial_epoch,
        backward_times,
        *,
        mirrored: int,
        chunks: tuple[list[list[float]], list[list[float]]],
        epoch_source,
        node_diagnostic,
    ) -> None:
        self._integrator = integrator
        self._system = system
        self._initial_epoch = initial_epoch
        self._backward_times = backward_times
        self._mirrored = mirrored
        self._forward_chunks, self._backward_chunks = chunks
        self._source = epoch_source
        self._node_diagnostic = node_diagnostic
        self._context = multiprocessing.get_context("fork")
        self._results: Connection | None
        self._results_writer: Connection | None
        self._results, self._results_writer = _pipe(self._context)
        self._batch_reader: Connection | None = None
        self._batches: Connection | None = None
        if epoch_source is not None:
            self._batch_reader, self._batches = _pipe(self._context)
        self._stash: list[tuple[list[float], object]] = []
        self._relayed: deque[tuple[list[float], object]] = deque()
        self._tags: Iterator[tuple[bool, list[float]]] = iter(())
        self._merged: Iterator | None = None
        self._process: BaseProcess | None = None
        self._start_error: OSError | None = None
        self._done: tuple[int, float, int] | None = None
        self._mirror_nodes: LunarNodeBuffer | None = None
        self._mirror_history: DelayedHistory | None = None
        self._diagnostic_queries = 0
        self._diagnostic_max = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def forward_source(self):
        """Epoch source of the forward main grid: this relay, or ``None`` to build the data in process."""
        return None if self._source is None else self

    def stream(self, initial_epoch, offset_batches):
        """Yield the forward batches of the interleaved stream, relaying the backward ones to the child."""
        forward = [list(offsets) for offsets in offset_batches]
        if forward != self._forward_chunks:
            raise RuntimeError("The forward main grid requested unexpected epoch batches")
        tagged = []
        for forward_offsets, backward_offsets in zip_longest(self._forward_chunks, self._backward_chunks):
            if forward_offsets is not None:
                tagged.append((True, forward_offsets))
            if backward_offsets is not None:
                tagged.append((False, backward_offsets))
        self._tags = iter(tagged)
        self._merged = self._source.stream(initial_epoch, [offsets for _, offsets in tagged])
        for done in range(len(forward)):
            if self._start_error is not None:
                # No child: the backward arc fetches its own batches after the forward arc.
                yield from self._source.stream(initial_epoch, forward[done:])
                return
            while True:
                is_forward, offsets = next(self._tags)
                batch = next(self._merged)
                if is_forward:
                    break
                self._offer(offsets, batch)
            yield batch

    def after_forward_node(self, nodes, index: int) -> None:
        """Start the child once the mirrored forward nodes exist; afterwards collect its nodes."""
        if self._process is not None:
            self._receive(block=False)
        elif index == self._mirrored - 1 and self._start_error is None:
            self._start(nodes)

    def finish(self):
        """Wait for the backward arc and return its node buffer and history.

        Returns
        -------
        tuple of (LunarNodeBuffer, DelayedHistory) or None
            Backward nodes, from the mirrored forward nodes on, and the history of the backward arc with the
            counters a sequential run would have; ``None`` if the child could not be forked, in which case the
            backward arc must be propagated here, from the propagation's own epoch source.

        Raises
        ------
        RuntimeError
            If the backward-arc process never started, exited without finishing or lost nodes.
        Exception
            The exception raised by the backward arc, chained to its remote traceback.
        """
        if self._start_error is not None:
            return None
        if self._process is None:
            raise RuntimeError("The backward arc was never started")
        # Each send waits while the child's pipe is full, so the relay keeps pace with the child.
        while self._relay_next():
            self._receive(block=False)
        while self._done is None:
            self._receive(block=True)
        self._process.join()
        queries, extrapolation, count = self._done
        nodes, history = self._mirror_nodes, self._mirror_history
        assert nodes is not None and history is not None
        if count != len(self._backward_times) - 1 or nodes.size != self._mirrored - 1 + len(self._backward_times):
            raise RuntimeError("The backward-arc process did not report every node")
        history.extrapolated_translation_queries = queries + self._diagnostic_queries
        history.max_translation_extrapolation_s = max(extrapolation, self._diagnostic_max)
        return nodes, history

    def close(self) -> None:
        """Stop the child if it is still running and release the process, the pipes and the epoch stream."""
        process, self._process = self._process, None
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(5.0)
                if process.is_alive():
                    process.kill()
            process.join()
            process.close()
        for end in (self._results, self._results_writer, self._batch_reader, self._batches):
            if end is not None:
                end.close()
        self._abandon_relay()

    def _abandon_relay(self) -> None:
        merged, self._merged = self._merged, None
        self._tags, self._stash = iter(()), []
        self._relayed.clear()
        if merged is not None and hasattr(merged, "close"):
            # An MPI stream drains its outstanding tasks when closed.
            merged.close()

    def _start(self, forward_nodes) -> None:
        stash, self._stash = self._stash, []
        process = self._context.Process(
            target=self._child_main,
            args=(forward_nodes, stash, os.getpid()),
            name="lunarops-backward-arc",
            daemon=True,
        )
        # An interrupt between the fork and recording the process would leave a child nobody stops.
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            # Garbage left by this process (an abandoned MPI stream, say) must not be finalised in the child.
            gc.freeze()
            try:
                process.start()
            except OSError as error:
                self._start_error = error
                self._abandon_relay()
                warnings.warn(
                    f"Could not fork the backward arc ({error}); it is propagated after the forward arc",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            finally:
                gc.unfreeze()
            self._process = process
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        self._relayed.extend(stash)
        assert self._results_writer is not None
        # Only the child writes results and reads batches: once it has gone, reading meets the end of the pipe
        # and writing raises, instead of waiting.
        self._results_writer.close()
        self._results_writer = None
        if self._batch_reader is not None:
            self._batch_reader.close()
            self._batch_reader = None
        self._mirror_nodes, self._mirror_history = self._integrator._backward_start(
            self._system, self._initial_epoch, forward_nodes, self._backward_times
        )

    def _child_main(self, forward_nodes, stash, parent_pid) -> None:
        # The parent decides on interruption and stops this process.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
        _die_with_parent(parent_pid)
        assert self._results is not None and self._results_writer is not None
        self._results.close()
        if self._batches is not None:
            self._batches.close()
        sender = _Sender(self._results_writer)
        try:
            nodes, history = self._integrator._backward_start(
                self._system, self._initial_epoch, forward_nodes, self._backward_times
            )
            reporter = _NodeReporter(
                sender, nodes, send_batches=self._node_diagnostic is not None and self._source is None
            )
            self._integrator._main_grid(
                self._system,
                self._initial_epoch,
                self._backward_times,
                nodes,
                history,
                prefetch=None,
                epoch_source=None if self._batch_reader is None else _RelayedEpochSource(stash, self._batch_reader),
                node_diagnostic=reporter,
                dop853_steps=0,
            )
            reporter.flush()
            count = nodes.size - nodes.trajectory_start - 1
            sender.send(
                ("done", history.extrapolated_translation_queries, history.max_translation_extrapolation_s, count)
            )
        except BaseException as error:  # noqa: BLE001 - every failure must reach the parent
            sender.send(("error", _portable(error), traceback.format_exc()))
        finally:
            # Closed first: a parent waiting to send the next batch then fails and turns to read our messages,
            # instead of both sides waiting on a full pipe.
            if self._batch_reader is not None:
                self._batch_reader.close()
            sender.close()

    def _offer(self, offsets, batch) -> None:
        if self._process is None:
            if self._start_error is None:
                self._stash.append((offsets, batch))
            return
        self._relayed.append((offsets, batch))
        assert self._batches is not None
        try:
            self._batches.send_bytes(pickle.dumps((offsets, batch), protocol=pickle.HIGHEST_PROTOCOL))
        except OSError:
            # The child has gone or stopped reading; its own message, if any, says why.
            while self._done is None:
                self._receive(block=True)
            raise RuntimeError("The backward-arc process stopped reading epoch batches") from None

    def _relay_next(self) -> bool:
        tagged = next(self._tags, None)
        if tagged is None:
            return False
        is_forward, offsets = tagged
        if is_forward:
            raise RuntimeError("Forward epoch batches remain after the forward arc")
        assert self._merged is not None
        self._offer(offsets, next(self._merged))
        return True

    def _receive(self, *, block: bool) -> None:
        """Handle the waiting child messages; when blocking, wait for one."""
        assert self._results is not None and self._process is not None
        while self._done is None:
            if not block and not self._results.poll():
                return
            try:
                data = self._results.recv_bytes()
            except EOFError, OSError:
                self._process.join(5.0)
                raise RuntimeError(
                    f"The backward-arc process exited with code {self._process.exitcode} before finishing"
                ) from None
            self._handle(pickle.loads(data))
            if block:
                return

    def _handle(self, message) -> None:
        kind = message[0]
        if kind == "done":
            self._done = message[1:]
            return
        if kind == "error":
            _, error, text = message
            error.__cause__ = _RemoteTraceback(text)
            raise error
        _, (times, states, derivatives, states_low, derivatives_low), slots, counters, batch = message
        nodes, history = self._mirror_nodes, self._mirror_history
        assert nodes is not None and history is not None
        first = nodes.size - self._mirrored + 1
        if not np.array_equal(times, self._backward_times[first : first + len(times)]):
            raise RuntimeError("The backward-arc process sent nodes out of order")
        if self._source is not None:
            relayed, batch = self._relayed.popleft()
            if relayed != [float(offset) for offset in times]:
                raise RuntimeError("Backward nodes do not match the relayed epoch batch")
        for column, slot in enumerate(slots):
            offset = float(times[column])
            state = np.ascontiguousarray(states[:, column])
            nodes.append(offset, state, derivatives[:, column], states_low[:, column], derivatives_low[:, column])
            if self._node_diagnostic is None:
                continue
            # The counters as a sequential run would show them: the child's queries up to this node plus the
            # diagnostics' own queries so far.
            queries, extrapolation = counters[column]
            base = queries + self._diagnostic_queries
            history.extrapolated_translation_queries = base
            history.max_translation_extrapolation_s = max(extrapolation, self._diagnostic_max)
            self._node_diagnostic(self._initial_epoch.shifted(offset), state, batch, slot, history)
            self._diagnostic_queries += history.extrapolated_translation_queries - base
            self._diagnostic_max = max(self._diagnostic_max, history.max_translation_extrapolation_s)
