"""The MPI runtime protocol, exercised through mpirun subprocesses."""

import json
import os
import subprocess
import sys
import time

import pytest

from tests.conftest import REPO_ROOT, mpi_launcher

_PUBLIC_API_RUNNER = """
import json
from types import SimpleNamespace
from lunarops.parallel import mpi

class Context:
    def __init__(self, cache):
        self.cache = cache
        self.closed = False
    def close(self):
        self.closed = True

contexts = []
def build(spec, cache):
    if spec["fail"]:
        raise ValueError("requested initialization failure")
    context = Context(cache)
    contexts.append(context)
    return context, SimpleNamespace(name=spec["name"])
mpi.build_worker_processor = build

def probe(payload, state):
    if payload["fail"]:
        raise ValueError("requested task failure")
    return state.processors[payload["specId"]].name, payload["value"]
mpi.TASK_HANDLERS["probe"] = probe
runtime = mpi.MpiRuntime()
if runtime.is_master:
    try:
        runtime.broadcast_spec("observationSpec", {"specId": "bad", "fail": True})
    except RuntimeError as exc:
        assert "requested initialization failure" in str(exc)
    else:
        raise AssertionError("initialization failure did not propagate")
    first = runtime.broadcast_spec("observationSpec", {"specId": "first", "fail": False, "name": "first"})
    assert first == runtime.broadcast_spec("observationSpec", {"specId": "first"})
    second = runtime.broadcast_spec("observationSpec", {"specId": "second", "fail": False, "name": "second"})
    assert first != second
    payloads = [{"specId": first, "value": i, "fail": False} for i in range(15)]
    expected = [("first", i) for i in range(15)]
    assert runtime.map_tasks("probe", payloads) == expected
    sizes = [2] * len(payloads)
    assert list(runtime.stream_tasks("probe", payloads, progress="probe", unit="points", sizes=sizes)) == expected
    assert runtime.map_tasks("probe", []) == []
    stream = runtime.stream_tasks("probe", payloads)
    assert next(stream) == expected[0]
    try:
        runtime.broadcast_spec("observationSpec", {"specId": "first"})
    except RuntimeError as exc:
        assert str(exc) == "broadcast_spec while a task stream is open"
    else:
        raise AssertionError("broadcast was allowed while a stream was open")
    stream.close()
    assert runtime.map_tasks("probe", payloads) == expected
    # A second stream is refused while one is open, and the open one keeps its results.
    stream = runtime.stream_tasks("probe", payloads)
    assert next(stream) == expected[0]
    try:
        runtime.stream_tasks("probe", payloads)
    except RuntimeError as exc:
        assert str(exc) == "stream_tasks while another task stream is open"
    else:
        raise AssertionError("a second stream was opened")
    assert list(stream) == expected[1:]
    # Of two streams created before either started, the one started second is refused.
    first_stream, second_stream = runtime.stream_tasks("probe", payloads), runtime.stream_tasks("probe", payloads)
    assert next(first_stream) == expected[0]
    try:
        next(second_stream)
    except RuntimeError as exc:
        assert str(exc) == "stream_tasks while another task stream is open"
    else:
        raise AssertionError("a second stream was started")
    assert list(first_stream) == expected[1:]
    broken = [dict(p, fail=i == 2) for i, p in enumerate(payloads)]
    try:
        runtime.map_tasks("probe", broken)
    except RuntimeError as exc:
        assert "requested task failure" in str(exc)
    else:
        raise AssertionError("task failure did not propagate")
    assert runtime.map_tasks("probe", [{"specId": second, "value": 99, "fail": False}]) == [("second", 99)]
    try:
        runtime.map_tasks("unknown_kind", [{}])
    except RuntimeError as exc:
        assert "unknown_kind" in str(exc)
    else:
        raise AssertionError("unknown task kind did not propagate")
    runtime.shutdown()
else:
    runtime.worker_loop()
closed = runtime.comm.gather((len(contexts), all(c.closed for c in contexts),
                             len({id(c.cache) for c in contexts})), root=0)
if runtime.is_master:
    assert closed[1:] == [(2, True, 1)] * (runtime.size - 1)
    print(json.dumps({"public_api_pass": True, "workers_closed": len(closed) - 1}), flush=True)
"""


@pytest.mark.mpi
def test_public_mpi_api_initialization_ordering_errors_drain_and_shutdown():
    result = subprocess.run(
        [mpi_launcher(), "-n", "3", sys.executable, "-c", _PUBLIC_API_RUNNER],
        cwd=REPO_ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary == {"public_api_pass": True, "workers_closed": 2}
    assert "observation model ready on 2 MPI workers" in result.stdout
    assert "probe  30/30 points (100%)  done in" in result.stdout


_INTERRUPT_RUNNER = """
import faulthandler
import json
from lunarops.parallel import mpi

faulthandler.dump_traceback_later(30, exit=True)

def probe(payload, state):
    return payload["value"]
mpi.TASK_HANDLERS["probe"] = probe

class Interrupting:
    # A Ctrl-C handled right after a receive or a send returned, before the stream could record it.
    def __init__(self, comm):
        self._comm, self.method, self.trips = comm, None, []
    def __getattr__(self, name):
        return getattr(self._comm, name)
    def _after(self, method, value):
        # trips: interrupt at the n-th call of method, then at the m-th call after that, ...
        if method == self.method and self.trips:
            self.trips[0] -= 1
            if not self.trips[0]:
                self.trips.pop(0)
                raise KeyboardInterrupt(method)
        return value
    def recv(self, *args, **kwargs):
        return self._after("recv", self._comm.recv(*args, **kwargs))
    def Isend(self, *args, **kwargs):
        # The request of the started send is lost, as by an interrupt before the runtime records it.
        return self._after("Isend", self._comm.Isend(*args, **kwargs))

runtime = mpi.MpiRuntime()
if runtime.is_master:
    comm = runtime.comm = Interrupting(runtime.comm)
    outcomes = {}
    # An interrupted stream leaves the runtime unsettled, and the next stream flushes the workers first. The
    # flush cases interrupt the stream and then the flush (its first receive or send) of the next stream.
    cases = (("recv", "recv", [12]), ("Isend", "Isend", [12]), ("flush", "recv", [12, 1]),
             ("flush-send", "Isend", [12, 1]))
    for case, method, trips in cases:
        comm.method, comm.trips = method, trips
        interrupts = 0
        while comm.trips:
            try:
                runtime.map_tasks("probe", [{"value": i} for i in range(40)])
            except KeyboardInterrupt:
                interrupts += 1
            else:
                raise AssertionError("no interrupt")
        # The next stream neither hangs nor receives a result of an interrupted one.
        outcomes[case] = [interrupts, runtime.map_tasks("probe", [{"value": 1000 + i} for i in range(40)])]
    runtime.shutdown()
    print(json.dumps(outcomes), flush=True)
else:
    runtime.worker_loop()
"""


@pytest.mark.mpi
def test_an_interrupted_stream_leaves_the_runtime_usable():
    result = subprocess.run(
        [mpi_launcher(), "-n", "3", sys.executable, "-c", _INTERRUPT_RUNNER],
        cwd=REPO_ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = list(range(1000, 1040))
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "recv": [1, expected],
        "Isend": [1, expected],
        "flush": [2, expected],
        "flush-send": [2, expected],
    }


_ABORT_RUNNER = """
import faulthandler
import sys
import time
from lunarops.parallel import mpi

# A regression hangs; every rank then exits after this delay, well after the test's time limit.
faulthandler.dump_traceback_later(20, exit=True)

def probe(payload, state):
    if payload["value"]:
        time.sleep(60)
    return payload["value"]
mpi.TASK_HANDLERS["probe"] = probe

class Interrupting:
    # Raises KeyboardInterrupt right after the n-th call of one method returns.
    def __init__(self, comm, method, n):
        self._comm, self.method, self.n = comm, method, n
    def __getattr__(self, name):
        attribute = getattr(self._comm, name)
        if name != self.method:
            return attribute
        def call(*args, **kwargs):
            value = attribute(*args, **kwargs)
            self.n -= 1
            if not self.n:
                raise KeyboardInterrupt(name)
            return value
        return call

case = sys.argv[1]
runtime = mpi.MpiRuntime()
if runtime.is_master:
    spec = {"specId": "probe", "fail": False}
    if case == "broadcast-send":
        # The first worker waits in the broadcast, the others for a message.
        runtime.comm = Interrupting(runtime.comm, "send", 1)
        runtime.broadcast_spec("epochInputs", spec)
    elif case == "broadcast-gather":
        # Every worker waits in the gather.
        runtime.comm = Interrupting(runtime.comm, "bcast", 1)
        runtime.broadcast_spec("epochInputs", spec)
    elif case == "shutdown":
        # A second Ctrl-C stops the flush that shutdown starts after an interrupted stream.
        runtime.comm = Interrupting(runtime.comm, "recv", 3)
        try:
            runtime.map_tasks("probe", [{"value": 0} for _ in range(20)])
        except KeyboardInterrupt:
            runtime.comm.n = 1
        runtime.shutdown()
    else:
        # Ctrl-C while every worker runs a long task: rank 0 does not wait for them (as in cli.cmd_run).
        try:
            for result in runtime.stream_tasks("probe", [{"value": i} for i in range(20)]):
                raise KeyboardInterrupt
        except BaseException as exc:
            runtime.shutdown(interrupted=mpi.is_interrupt(exc))
            raise
    print("rank 0 was not aborted", flush=True)
else:
    mpi.WorkerState.initialize = lambda self, kind, spec_id, spec: None
    runtime.worker_loop()
"""


@pytest.mark.mpi
@pytest.mark.parametrize("case", ["broadcast-send", "broadcast-gather", "shutdown", "interrupted-run"])
def test_an_interrupt_that_cannot_be_settled_aborts_the_job_promptly(case):
    """Interrupts inside a spec broadcast, inside shutdown or of the run itself end the job with MPI_Abort."""
    started = time.monotonic()
    result = subprocess.run(
        [mpi_launcher(), "-n", "3", sys.executable, "-c", _ABORT_RUNNER, case],
        cwd=REPO_ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    elapsed = time.monotonic() - started
    assert result.returncode != 0, result.stdout + result.stderr
    assert elapsed < 15, result.stdout + result.stderr
    assert "aborting the MPI job" in result.stderr, result.stdout + result.stderr
    assert "rank 0 was not aborted" not in result.stdout


_SIGINT_RUNNER = """
import faulthandler
import json
import os
import signal
import time
from lunarops.parallel import mpi

faulthandler.dump_traceback_later(30, exit=True)

def probe(payload, state):
    if payload["interrupt"]:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
    return payload["value"]
mpi.TASK_HANDLERS["probe"] = probe

runtime = mpi.MpiRuntime()
pids = runtime.comm.gather(os.getpid(), root=0)
if runtime.is_master:
    # Once the workers have answered a task, they are in their loop.
    assert runtime.map_tasks("probe", [{"value": 0, "interrupt": False}] * 4) == [0] * 4
    # SIGINT while the workers wait for a message, then during their tasks.
    for pid in pids[1:]:
        os.kill(pid, signal.SIGINT)
    time.sleep(0.2)
    values = runtime.map_tasks("probe", [{"value": i, "interrupt": i % 3 == 0} for i in range(20)])
    runtime.shutdown()
else:
    runtime.worker_loop()
restored = runtime.comm.gather(signal.getsignal(signal.SIGINT) is signal.default_int_handler, root=0)
if runtime.is_master:
    print(json.dumps({"values": values, "restored": restored}), flush=True)
"""


@pytest.mark.mpi
def test_workers_ignore_sigint_and_leave_stopping_to_rank_0():
    result = subprocess.run(
        [mpi_launcher(), "-n", "3", sys.executable, "-c", _SIGINT_RUNNER],
        cwd=REPO_ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"values": list(range(20)), "restored": [True] * 3}


_FAILURE_RUNNER = """
import sys
import numpy as np
from lunarops.parallel import mpi

def large_result(payload, state):
    return np.full(1_000_000, payload["taskId"], dtype=float)
mpi.TASK_HANDLERS["large"] = large_result
runtime = mpi.MpiRuntime()
if runtime.is_master:
    try:
        if sys.argv[1] == "consumer":
            stream = runtime.stream_tasks("large", [{} for _ in range(24)])
            for index, result in enumerate(stream):
                if index == 1:
                    raise ValueError("requested consumer failure")
        else:
            runtime.comm.send(("unknown_kind", {}), dest=1, tag=mpi.TASK)
            runtime.comm.recv(source=1, tag=mpi.RESULT)
    finally:
        runtime.shutdown()
else:
    runtime.worker_loop()
"""


@pytest.mark.mpi
@pytest.mark.slow
@pytest.mark.parametrize("failure", ["consumer", "malformed"])
def test_mpi_failure_exits_without_hanging(failure):
    """A consumer error or an undecodable task ends the job with a non-zero status instead of hanging."""
    result = subprocess.run(
        [mpi_launcher(), "-n", "4", sys.executable, "-c", _FAILURE_RUNNER, failure],
        cwd=REPO_ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    if failure == "consumer":
        assert "requested consumer failure" in result.stderr
