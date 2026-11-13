"""Worker initializers for ProcessRuntime tests.

Spawned workers start from a fresh interpreter, so a monkeypatch in the test process does not reach them:
each test runtime passes one of these module-level functions as its ``initializer`` instead.
"""

import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lunarops.parallel import observation_spec, tasks

CLOSED_DIR_VARIABLE = "LUNAROPS_TEST_CLOSED_DIR"
"""Directory in which probe contexts record that they were closed (one file per context)."""


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


def install():
    """Install the probe handlers and a probe observation-spec builder."""
    tasks.build_worker_processor = _build_probe
    tasks.TASK_HANDLERS.update(probe=_probe, large=_large, pid=_pid)


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
