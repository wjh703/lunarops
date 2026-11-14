"""The local process runtime: the MPI runtime protocol on spawned worker processes."""

import json
import multiprocessing
import multiprocessing.context
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import nullcontext

import numpy as np
import pytest

from lunarops.parallel import process
from lunarops.parallel.process import ProcessRuntime
from lunarops.parallel.runtime import TaskRuntime, Terminated, is_interrupt, starting_children
from tests.conftest import REPO_ROOT
from tests.parallel import probe_handlers
from tests.parallel.probe_handlers import THREAD_VARIABLES


class _Hang(Exception):
    """Raised by the alarm when a test waits far longer than it should."""


@pytest.fixture(autouse=True)
def _guarded():
    # A regression must fail the test, not hang the suite; no worker may outlive its test.
    def hung(signum, frame):
        raise _Hang("the test hung")

    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(60)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert multiprocessing.active_children() == []


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv(probe_handlers.CLOSED_DIR_VARIABLE, str(tmp_path))
    runtime = ProcessRuntime(2, initializer=probe_handlers.install)
    yield runtime
    runtime.shutdown()


def _spec(name, *, fail=False):
    return {"specId": name, "fail": fail, "name": name}


def _payloads(spec_id, count=15, **options):
    return [{"specId": spec_id, "value": i, **options} for i in range(count)]


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_the_runtime_implements_the_task_runtime_interface(runtime):
    assert isinstance(runtime, TaskRuntime)
    assert (runtime.name, runtime.workers, runtime.describe()) == ("process", 2, "2 worker processes")
    assert len(set(runtime.pids)) == 2 and os.getpid() not in runtime.pids


def test_results_keep_task_order_and_specs_are_built_once(runtime, tmp_path, capsys):
    with pytest.raises(RuntimeError, match="requested initialization failure"):
        runtime.broadcast_spec("observationSpec", _spec("bad", fail=True))
    first = runtime.broadcast_spec("observationSpec", _spec("first"))
    assert first == runtime.broadcast_spec("observationSpec", {"specId": "first"})
    second = runtime.broadcast_spec("observationSpec", _spec("second"))
    assert first != second
    expected = [("first", i) for i in range(15)]
    assert runtime.map_tasks("probe", _payloads(first)) == expected
    stream = runtime.stream_tasks("probe", _payloads(first), progress="probe", unit="points", sizes=[2] * 15)
    assert list(stream) == expected
    assert runtime.map_tasks("probe", []) == []
    assert runtime.map_tasks("probe", _payloads(second, 1)) == [("second", 0)]
    output = capsys.readouterr().out
    assert "observation model ready on 2 worker processes" in output
    assert "probe  30/30 points (100%)  done in" in output
    runtime.shutdown()
    # Every worker closed its two contexts, which share the worker's class cache.
    closed = [name.split("-") for name in os.listdir(tmp_path)]
    assert sorted(name for _, name, _ in closed) == ["first", "first", "second", "second"]
    assert len({(pid, cache) for pid, _, cache in closed}) == 2


def test_streams_and_broadcasts_are_refused_while_a_stream_is_open(runtime):
    spec_id = runtime.broadcast_spec("observationSpec", _spec("first"))
    expected = [("first", i) for i in range(15)]
    stream = runtime.stream_tasks("probe", _payloads(spec_id))
    assert next(stream) == expected[0]
    with pytest.raises(RuntimeError, match="^broadcast_spec while a task stream is open$"):
        runtime.broadcast_spec("observationSpec", {"specId": "first"})
    with pytest.raises(RuntimeError, match="^stream_tasks while another task stream is open$"):
        runtime.stream_tasks("probe", _payloads(spec_id))
    with pytest.raises(RuntimeError, match="^limit while a task stream is open$"), runtime.limit(1):
        pass
    assert list(stream) == expected[1:]
    # Of two streams created before either started, the one started second is refused.
    first_stream, second_stream = (
        runtime.stream_tasks("probe", _payloads(spec_id)),
        runtime.stream_tasks("probe", _payloads(spec_id)),
    )
    assert next(first_stream) == expected[0]
    with pytest.raises(RuntimeError, match="^stream_tasks while another task stream is open$"):
        next(second_stream)
    assert list(first_stream) == expected[1:]


def test_a_stream_closed_early_leaves_no_result_behind(runtime):
    stream = runtime.stream_tasks("probe", _payloads(None, 40))
    assert next(stream) == (None, 0)
    stream.close()
    assert runtime.map_tasks("probe", [{"value": 1000 + i} for i in range(40)]) == [(None, 1000 + i) for i in range(40)]


def test_task_failures_are_reported_and_the_runtime_recovers(runtime):
    broken = [dict(payload, fail=i == 2) for i, payload in enumerate(_payloads(None))]
    with pytest.raises(RuntimeError, match=r"worker process \d \(pid \d+\) failed on probe task 2:\n(.|\n)*requested"):
        runtime.map_tasks("probe", broken)
    with pytest.raises(RuntimeError, match="KeyError: 'unknown_kind'"):
        runtime.map_tasks("unknown_kind", [{}])
    assert runtime.map_tasks("probe", _payloads(None, 3)) == [(None, 0), (None, 1), (None, 2)]


def test_a_consumer_error_leaves_the_runtime_usable(runtime):
    def consume():
        stream = runtime.stream_tasks("large", [{} for _ in range(24)])
        try:
            for index, _ in enumerate(stream):
                if index == 1:
                    raise ValueError("requested consumer failure")
        finally:
            stream.close()

    with pytest.raises(ValueError, match="requested consumer failure"):
        consume()
    assert runtime.map_tasks("probe", _payloads(None, 2)) == [(None, 0), (None, 1)]


def test_large_tasks_and_results_cross_without_waiting_on_each_other(runtime):
    # While a worker sends an 8 MB result it receives its next 8 MB task: neither side may wait for the other.
    payload = {"padding": np.zeros(1_000_000)}
    results = runtime.map_tasks("large", [payload] * 8)
    assert [result[0] for result in results] == list(range(8)) and all(len(r) == 1_000_000 for r in results)


def test_limit_restricts_the_tasks_to_the_first_workers(runtime):
    with runtime.limit(1):
        assert set(runtime.map_tasks("pid", [{}] * 6)) == {runtime.pids[0]}
    assert set(runtime.map_tasks("pid", [{"sleep": 0.05}] * 12)) == set(runtime.pids)
    with pytest.raises(ValueError, match="between 1 and 2"), runtime.limit(3):
        pass


def test_an_initializer_failure_is_reported_and_no_worker_is_left():
    with pytest.raises(RuntimeError, match=r"failed to run its initializer:\n(.|\n)*requested initializer failure"):
        ProcessRuntime(2, initializer=probe_handlers.install_and_fail)


def test_workers_ignore_sigint_and_leave_stopping_to_the_master(runtime):
    assert runtime.map_tasks("probe", _payloads(None, 4)) == [(None, i) for i in range(4)]
    for pid in runtime.pids:
        os.kill(pid, signal.SIGINT)
    time.sleep(0.2)
    values = runtime.map_tasks("probe", [{"value": i, "interrupt": i % 3 == 0} for i in range(20)])
    assert values == [(None, i) for i in range(20)]


def test_a_worker_that_dies_is_reported_promptly_and_breaks_the_runtime(runtime):
    started = time.monotonic()
    payloads = [{"value": i, "exit": 3 if i == 5 else None, "sleep": 0.01} for i in range(40)]
    with pytest.raises(RuntimeError, match=r"worker process \d \(pid \d+\) exited with code 3 during probe tasks"):
        runtime.map_tasks("probe", payloads)
    assert time.monotonic() - started < 10
    pids = runtime.pids
    with pytest.raises(RuntimeError, match="worker processes were stopped"):
        runtime.map_tasks("probe", payloads)
    with pytest.raises(RuntimeError, match="worker processes were stopped"):
        runtime.broadcast_spec("observationSpec", _spec("first"))
    assert not any(_alive(pid) for pid in pids)


def test_a_killed_worker_is_reported_with_its_signal(runtime):
    os.kill(runtime.pids[1], signal.SIGKILL)
    with pytest.raises(RuntimeError, match=r"worker process 2 \(pid \d+\) was killed by SIGKILL"):
        runtime.map_tasks("probe", [{"value": i, "sleep": 0.05} for i in range(20)])


def test_an_interrupt_inside_a_stream_stops_the_workers_at_once(runtime):
    # Ctrl-C while the master waits for results that the workers will not send for a minute.
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGUSR1, interrupt)
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGUSR1))
    started = time.monotonic()
    try:
        stream = runtime.stream_tasks("probe", [{"value": i, "sleep": 60} for i in range(8)])
        timer.start()
        with pytest.raises(KeyboardInterrupt):
            next(stream)
    finally:
        timer.join()
        signal.signal(signal.SIGUSR1, previous)
    assert time.monotonic() - started < 15
    assert not any(_alive(pid) for pid in runtime.pids)
    with pytest.raises(RuntimeError, match="KeyboardInterrupt during probe tasks"):
        runtime.map_tasks("probe", [])


def test_terminated_is_an_interrupt():
    assert is_interrupt(Terminated()) and is_interrupt(KeyboardInterrupt())
    assert not any(is_interrupt(exc) for exc in (ValueError(), GeneratorExit(), SystemExit(1)))
    assert Terminated(signal.SIGTERM).exit_code == 143


def test_an_exception_in_the_middle_of_a_received_message_breaks_the_runtime(runtime):
    # An ordinary exception after the header of a result was read: draining the out-of-step pipe would read the
    # result's bytes as the next header.
    class Injected(Exception):
        pass

    armed = [True]
    for worker in runtime._workers:

        def recv(size, read=worker.conn._recv):
            data = read(size)
            if size == 4 and armed[0]:
                armed[0] = False
                raise Injected
            return data

        worker.conn._recv = recv
    started = time.monotonic()
    with pytest.raises(Injected):
        runtime.map_tasks("probe", [{"value": i, "sleep": 0.01} for i in range(8)])
    assert time.monotonic() - started < 10
    assert not armed[0]
    assert not any(_alive(pid) for pid in runtime.pids)
    with pytest.raises(RuntimeError, match=r"stopped \(Injected during probe tasks\)"):
        runtime.map_tasks("probe", [])


def _pending_stream(runtime):
    """Return a stream that has yielded its first result while the workers hold tasks of a minute."""
    stream = runtime.stream_tasks("probe", [{"value": i, "sleep": 60 if i else 0} for i in range(8)])
    assert next(stream) == (None, 0)
    return stream


def test_a_drain_that_gets_no_answer_stops_the_workers_after_its_deadline(runtime, monkeypatch, capsys):
    monkeypatch.setattr(process, "_DRAIN_TIMEOUT_S", 0.5)
    stream = _pending_stream(runtime)
    started = time.monotonic()
    stream.close()
    assert time.monotonic() - started < 5
    assert not any(_alive(pid) for pid in runtime.pids)
    assert "did not return the results of the stopped probe tasks within 0.5 s" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match=r"stopped \(results of stopped probe tasks not returned within 0.5 s\)"):
        runtime.map_tasks("probe", [])


def test_an_interrupt_during_a_drain_stops_the_workers(runtime):
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    stream = _pending_stream(runtime)
    previous = signal.signal(signal.SIGUSR1, interrupt)
    timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGUSR1))
    started = time.monotonic()
    try:
        timer.start()
        with pytest.raises(KeyboardInterrupt):
            stream.close()
    finally:
        timer.join()
        signal.signal(signal.SIGUSR1, previous)
    assert time.monotonic() - started < 5
    assert not any(_alive(pid) for pid in runtime.pids)
    with pytest.raises(RuntimeError, match="KeyboardInterrupt while draining probe tasks"):
        runtime.map_tasks("probe", [])


@pytest.mark.parametrize("guarded", [True, False], ids=["starting_children", "unguarded"])
def test_children_started_inside_starting_children_ignore_sigint_from_their_first_instruction(guarded, monkeypatch):
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    monkeypatch.delenv("MKL_NUM_THREADS")
    handler = signal.getsignal(signal.SIGINT)
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=probe_handlers.report_start, args=(writer,))
    with starting_children() if guarded else nullcontext():
        child.start()
    writer.close()
    try:
        assert reader.poll(30)
        state = reader.recv()
    finally:
        child.join(30)
        reader.close()
    unguarded = {**dict.fromkeys(THREAD_VARIABLES, "8"), "MKL_NUM_THREADS": None}
    threads = dict.fromkeys(THREAD_VARIABLES, "1") if guarded else unguarded
    assert state == {"sigintIgnored": guarded, "threads": threads}
    # This process is left as it was.
    assert signal.getsignal(signal.SIGINT) is handler
    assert [os.environ.get(name) for name in THREAD_VARIABLES] == ["8", "8", None]


def _spy_on_spawns(monkeypatch):
    """Record the SIGINT handler and the thread variables of this process whenever it spawns a process."""
    seen = []
    start = multiprocessing.context.SpawnProcess.start

    def spy(self):
        seen.append((signal.getsignal(signal.SIGINT), {name: os.environ.get(name) for name in THREAD_VARIABLES}))
        start(self)

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", spy)
    return seen


def test_workers_start_with_sigint_ignored_and_one_native_thread(monkeypatch):
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    seen = _spy_on_spawns(monkeypatch)
    handler = signal.getsignal(signal.SIGINT)
    with ProcessRuntime(2, initializer=probe_handlers.install) as runtime:
        states = runtime.map_tasks("start_state", [{}, {}])
    assert seen == [(signal.SIG_IGN, dict.fromkeys(THREAD_VARIABLES, "1"))] * 2
    assert states == [{"sigintIgnored": True, "threads": dict.fromkeys(THREAD_VARIABLES, "1")}] * 2
    assert all(os.environ[name] == "8" for name in THREAD_VARIABLES)
    assert signal.getsignal(signal.SIGINT) is handler


def test_an_interrupted_run_does_not_wait_for_the_tasks_the_workers_hold(runtime):
    started = time.monotonic()
    try:
        for _ in runtime.stream_tasks("probe", [{"value": i, "sleep": 0 if i < 2 else 60} for i in range(8)]):
            raise KeyboardInterrupt
    except BaseException as exc:  # noqa: BLE001 -- as cli.cmd_run does
        runtime.shutdown(interrupted=is_interrupt(exc))
    assert time.monotonic() - started < 15
    assert multiprocessing.active_children() == []
    runtime.shutdown()  # a second call does nothing


def test_shutdown_stops_the_workers_and_refuses_further_work(runtime):
    pids = runtime.pids
    runtime.shutdown()
    assert not any(_alive(pid) for pid in pids)
    with pytest.raises(RuntimeError, match="shut down"):
        runtime.map_tasks("probe", [])
    runtime.shutdown()


def test_the_context_manager_shuts_the_workers_down():
    with ProcessRuntime(1, initializer=probe_handlers.install) as runtime:
        assert runtime.describe() == "1 worker process"
        assert runtime.map_tasks("probe", [{"value": 1}]) == [(None, 1)]
    assert not any(_alive(pid) for pid in runtime.pids)


@pytest.mark.parametrize("workers", [0, -1])
def test_at_least_one_worker_is_required(workers):
    with pytest.raises(ValueError, match="at least one worker"):
        ProcessRuntime(workers)


_ORPHAN_RUNNER = """
import json
import os
import signal
import sys
import threading
import time
from lunarops.parallel.process import ProcessRuntime
from tests.parallel import probe_handlers

runtime = ProcessRuntime(2, initializer=probe_handlers.install)
print(json.dumps(runtime.pids), flush=True)
if sys.argv[1] == "idle":
    os.kill(os.getpid(), signal.SIGKILL)
# The workers are busy with long tasks when this process dies.
threading.Thread(target=lambda: (time.sleep(1.0), os.kill(os.getpid(), signal.SIGKILL)), daemon=True).start()
runtime.map_tasks("probe", [{"value": i, "sleep": 60} for i in range(4)])
"""


@pytest.mark.parametrize("case", ["idle", "busy"])
def test_workers_exit_when_the_master_dies(case):
    result = subprocess.run(
        [sys.executable, "-c", _ORPHAN_RUNNER, case],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == -signal.SIGKILL, result.stdout + result.stderr
    # The workers exit without a traceback, also when a reply meets the broken pipe.
    assert "Traceback" not in result.stderr, result.stderr
    pids = json.loads(result.stdout.splitlines()[0])
    deadline = time.monotonic() + 10
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(_alive(pid) for pid in pids)
