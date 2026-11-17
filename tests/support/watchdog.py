"""Watchdog for tests that start worker processes: a hang fails the test, and no worker outlives it."""

import multiprocessing
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class Hung(Exception):
    """Raised by the watchdog alarm; not an ``OSError`` or ``RuntimeError``, which the code under test may handle."""


@contextmanager
def watchdog(seconds: int, *, single_thread: bool = False) -> Iterator[None]:
    """Fail a block that runs longer than ``seconds``, then check that it left no child process behind.

    Parameters
    ----------
    seconds : int
        ``SIGALRM`` timeout; ``Hung`` is raised in the main thread when it expires.
    single_thread : bool, default False
        Also check that only the main thread is left.
    """

    def hung(signum, frame):
        raise Hung("the test hung")

    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert multiprocessing.active_children() == []
    if single_thread:
        assert threading.active_count() == 1, threading.enumerate()
