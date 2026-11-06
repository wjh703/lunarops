"""The MPI runtime protocol, exercised through mpirun subprocesses."""

import json
import os
import subprocess
import sys

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
        cwd=REPO_ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary == {"public_api_pass": True, "workers_closed": 2}
    assert "observation model ready on 2 MPI workers" in result.stdout
    assert "probe  30/30 points (100%)  done in" in result.stdout


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
        cwd=REPO_ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode != 0
    if failure == "consumer":
        assert "requested consumer failure" in result.stderr
