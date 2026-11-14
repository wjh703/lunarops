"""Worker initializers for ProcessRuntime tests.

Spawned workers start from a fresh interpreter, so a monkeypatch in the test process does not reach them:
each test runtime passes one of these module-level functions as its ``initializer`` instead.
"""

import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lunarops.parallel import observation_spec, tasks

CLOSED_DIR_VARIABLE = "LUNAROPS_TEST_CLOSED_DIR"
"""Directory in which probe contexts record that they were closed (one file per context)."""

START_DIR_VARIABLE = "LUNAROPS_TEST_START_DIR"
"""Directory in which ``record_start`` writes the start-up state of each process (one file per process)."""

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def start_state() -> dict:
    """Return whether SIGINT is ignored and the native thread variables of this process."""
    return {
        "sigintIgnored": signal.getsignal(signal.SIGINT) == signal.SIG_IGN,
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def report_start(conn):
    """Process target: send ``start_state`` as the first thing the process does."""
    conn.send(start_state())
    conn.close()


class ProbeContext:
    def __init__(self, cache, name):
        self.cache, self.name = cache, name

    def close(self):
        directory = os.environ.get(CLOSED_DIR_VARIABLE)
        if directory:
            Path(directory, f"{os.getpid()}-{self.name}-{id(self.cache)}").touch()


def _build_probe(spec, cache):
    if spec["fail"]:
        raise ValueError("requested initialization failure")
    return ProbeContext(cache, spec["name"]), SimpleNamespace(name=spec["name"])


def _probe(payload, state):
    if payload.get("fail"):
        raise ValueError("requested task failure")
    if payload.get("exit") is not None:
        os._exit(payload["exit"])
    if payload.get("interrupt"):
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
    if payload.get("sleep"):
        time.sleep(payload["sleep"])
    name = None if payload.get("specId") is None else state.processors[payload["specId"]].name
    return name, payload["value"]


def _large(payload, state):
    return np.full(1_000_000, payload["taskId"], dtype=float)


def _pid(payload, state):
    return os.getpid()


def _start_state(payload, state):
    return start_state()


def install():
    """Install the probe handlers and a probe observation-spec builder."""
    tasks.build_worker_processor = _build_probe
    tasks.TASK_HANDLERS.update(probe=_probe, large=_large, pid=_pid, start_state=_start_state)


def install_and_fail():
    """Fail as an initializer would on a worker that cannot set up."""
    raise ValueError("requested initializer failure")


def _solar_system(config, context):
    from tests.support.dynamics import solar_system

    return solar_system()


def _circular_moon(config, context):
    from tests.support.dynamics import circular_moon_ephemeris

    return circular_moon_ephemeris()


def register_test_factories():
    """Register the analytic ephemeris types ``testAnalyticSolarSystem`` and ``testCircularMoon``."""
    from lunarops.config.registry import register_factory

    register_factory("ephemerides", "testAnalyticSolarSystem", _solar_system)
    register_factory("ephemerides", "testCircularMoon", _circular_moon)


def record_start_and_register():
    """Write ``start_state`` to ``$LUNAROPS_TEST_START_DIR/<pid>.json``, then ``register_test_factories``."""
    Path(os.environ[START_DIR_VARIABLE], f"{os.getpid()}.json").write_text(json.dumps(start_state()))
    register_test_factories()


def _static_context(spec, cache):
    from tests.support.observation import StaticKernelContext

    return StaticKernelContext(
        working_dir=spec.get("workingDir", "."), mpi_resources=spec.get("sharedResources"), class_cache=cache
    )


def _static_processor(spec, cache):
    from lunarops.classes.observation.assembly import build_observation_processor

    context = _static_context(spec, cache)
    processor = build_observation_processor(
        context,
        spec["programConfig"],
        station_catalog=spec["stationCatalog"],
        reflector_catalog=spec["reflectorCatalog"],
    )
    return context, processor


def _static_runtime(spec, cache):
    return observation_spec.build_worker_observation_runtime(spec, cache, context=_static_context(spec, cache))


def install_static_observation():
    """Build the observation models of the workers on the static system of ``StaticKernelContext``."""
    tasks.build_worker_processor = _static_processor
    tasks.build_worker_observation_runtime = _static_runtime
