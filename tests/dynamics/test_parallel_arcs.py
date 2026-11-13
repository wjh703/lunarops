"""Two-sided propagation with the backward arc in a spawned process."""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, arc_process
from lunarops.classes.dynamics.arc_process import BackwardArcProcess
from lunarops.classes.dynamics.epoch_data import SerialEpochSource
from lunarops.parallel.epoch_stream import RuntimeEpochSource
from lunarops.parallel.process import ProcessRuntime
from tests.conftest import REPO_ROOT, mpi_launcher
from tests.dynamics import arc_cases
from tests.dynamics.arc_cases import CASES, DAY, RecordingDiagnostic
from tests.parallel import probe_handlers
from tests.support.dynamics import EPOCH


class _Hung(Exception):
    pass


@pytest.fixture(autouse=True)
def _no_process_left_over():
    def hung(signum, frame):
        raise _Hung("the test hung")  # not an OSError or RuntimeError, which the arc handles

    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(120)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert multiprocessing.active_children() == []
    assert threading.active_count() == 1, threading.enumerate()


def _arc(factory, workers=0):
    return BackwardArcProcess(factory, workers=workers, initializer=probe_handlers.register_test_factories)


def _propagate(case, *, parallel, workers=0, factory=None, diagnostic=None, epoch_source=None, arcs=None):
    """Propagate a case two-sided; return the trajectory, the diagnostic calls and the progress counts."""
    settings, system, state, spans, options = CASES[case]()
    record = RecordingDiagnostic() if diagnostic is None else diagnostic
    record.bind(system)
    progress = []
    with ExitStack() as stack:
        arc = None
        if parallel:
            arc = stack.enter_context(_arc(factory or partial(arc_cases.setup, case), workers))
            if arcs is not None:
                arcs.append(arc)
            arc.start()
        trajectory = AdamsBashforthMoultonIntegrator(settings).integrate(
            system,
            EPOCH,
            state,
            spans["duration"],
            backward_duration_s=spans["back"],
            node_diagnostic=record,
            epoch_source=epoch_source,
            backward_arc=arc,
            progress=progress.append,
            **options,
        )
    return trajectory, record.calls, progress


def _assert_same_calls(got, expected):
    assert [call[0] for call in got] == [call[0] for call in expected]
    for (_, state, derivative, slot, counters), (_, state0, derivative0, slot0, counters0) in zip(
        got, expected, strict=True
    ):
        np.testing.assert_array_equal(state, state0)
        np.testing.assert_array_equal(derivative, derivative0)
        assert (slot, counters) == (slot0, counters0)


def _assert_identical(parallel, sequential):
    trajectory, calls, progress = parallel
    trajectory0, calls0, progress0 = sequential
    np.testing.assert_array_equal(trajectory.times_s, trajectory0.times_s)
    np.testing.assert_array_equal(trajectory.states, trajectory0.states)
    if trajectory0.history is not None:
        for arc in ("_forward", "_backward"):
            got, expected = getattr(trajectory.history, arc), getattr(trajectory0.history, arc)
            assert (got.extrapolated_translation_queries, got.max_translation_extrapolation_s) == (
                expected.extrapolated_translation_queries,
                expected.max_translation_extrapolation_s,
            ), arc
    # Every node is reported once, the forward arc first, with the slot and the history counters a sequential
    # run shows at that node.
    _assert_same_calls(calls, calls0)
    assert sum(progress) == sum(progress0) == len(trajectory0.times_s)


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_parallel_backward_arc_is_bitwise_identical_to_the_sequential_propagation(case):
    _assert_identical(_propagate(case, parallel=True), _propagate(case, parallel=False))


def test_the_cases_cover_one_and_many_mirrored_nodes():
    counts = {}
    for case, build in CASES.items():
        settings, system, *_ = build()
        counts[case] = AdamsBashforthMoultonIntegrator(settings)._mirrored_node_count(system)
    assert counts["single-mirror"] == 1
    assert counts["advanced-tide"] > 64  # seeded in the second forward epoch batch


def test_the_history_counters_are_exercised():
    sequential, _, _ = _propagate("tide-deformation", parallel=False)
    assert sequential.history._backward.extrapolated_translation_queries > 0
    assert sequential.history._forward.extrapolated_translation_queries > 0


def test_worker_processes_feed_both_arcs_bitwise():
    # The forward arc on one of two runtime workers, the backward arc on a worker of its own.
    _, system, *_ = CASES["advanced-tide"]()
    with ProcessRuntime(2, initializer=probe_handlers.register_test_factories) as runtime:
        source = RuntimeEpochSource(
            runtime, system.epoch_inputs, arc_cases.EPOCH_SOURCE_SPEC["ephemerides"], ".", "forward"
        )
        with runtime.limit(1):
            parallel = _propagate("advanced-tide", parallel=True, workers=1, epoch_source=source)
    sequential = _propagate(
        "advanced-tide", parallel=False, epoch_source=SerialEpochSource(system.epoch_inputs, system.ephemeris)
    )
    _assert_identical(parallel, sequential)


def test_a_backward_arc_needs_a_two_sided_propagation_and_an_arc_diagnostic():
    settings, system, state, spans, _ = CASES["two-body"]()
    integrator = AdamsBashforthMoultonIntegrator(settings)
    with _arc(partial(arc_cases.setup, "two-body")) as arc:
        with pytest.raises(ValueError, match="two-sided"):
            integrator.integrate(system, EPOCH, state, spans["duration"], backward_arc=arc)
        with pytest.raises(TypeError, match="ArcDiagnostic"):
            integrator.integrate(
                system,
                EPOCH,
                state,
                spans["duration"],
                backward_duration_s=spans["back"],
                node_diagnostic=lambda *args: None,
                backward_arc=arc,
            )
        assert arc.pid is None


def _fails(action, **kwargs):
    return partial(arc_cases.failing_setup, "two-body", action, **kwargs)


def test_an_error_on_the_backward_arc_is_raised_with_its_remote_traceback():
    with pytest.raises(ValueError, match="backward arc failed") as raised:
        _propagate("two-body", parallel=True, factory=_fails("raise"))
    assert "in _fail" in str(raised.value.__cause__)


def test_an_unpicklable_error_on_the_backward_arc_is_reported_by_name():
    with pytest.raises(RuntimeError, match="LocalError: not importable"):
        _propagate("two-body", parallel=True, factory=_fails("unpicklable"))


@pytest.mark.parametrize("workers", [0, 1])
def test_a_backward_process_that_dies_is_reported(workers):
    with pytest.raises(RuntimeError, match="exited with code 3"):
        _propagate("two-body", parallel=True, workers=workers, factory=_fails("exit"))


def test_a_failing_factory_is_reported():
    with pytest.raises(ValueError, match="cannot build two-body") as raised:
        _propagate("two-body", parallel=True, factory=partial(arc_cases.broken_setup, "two-body"))
    assert "in broken_setup" in str(raised.value.__cause__)


def test_a_backward_process_that_builds_another_problem_is_rejected():
    with pytest.raises(RuntimeError, match="different problem"):
        _propagate("two-body", parallel=True, factory=partial(arc_cases.mismatched_setup, "two-body"))


def test_a_stalled_backward_process_is_stopped(monkeypatch):
    monkeypatch.setattr(arc_process, "STALL_TIMEOUT_S", 2.0)
    arcs = []
    with pytest.raises(RuntimeError, match="sent nothing for 2 s"):
        _propagate("two-body", parallel=True, workers=1, factory=_fails("stall"), arcs=arcs)
    pids = [arcs[0].pid, *arcs[0].worker_pids]
    assert len(pids) == 2
    assert _gone(pids)


def _gone(pids, timeout_s=15.0):
    """Return whether every process has exited (zombies count as exited) within ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while True:
        alive = []
        for pid in pids:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except OSError:
                continue
            if state not in ("Z", "X"):
                alive.append(pid)
        if not alive or time.monotonic() > deadline:
            return not alive
        time.sleep(0.05)


class _FailingForward(RecordingDiagnostic):
    """Raises on the forward arc once the backward-arc process has reported its workers."""

    def __init__(self, error, arcs):
        super().__init__()
        self._error = error
        self._arcs = arcs

    def __call__(self, epoch, *args):
        super().__call__(epoch, *args)
        if EPOCH.seconds_until(epoch) > 3 * DAY:
            arc = self._arcs[0]
            while not arc.worker_pids:
                arc.poll()
                time.sleep(0.01)
            raise self._error


@pytest.mark.parametrize(
    "error", [KeyError("forward diagnostic failed"), KeyboardInterrupt()], ids=["error", "interrupt"]
)
def test_a_forward_failure_stops_the_backward_process_and_its_workers(error):
    arcs = []
    with pytest.raises(type(error)):
        _propagate("two-body", parallel=True, workers=1, diagnostic=_FailingForward(error, arcs), arcs=arcs)
    pids = [arcs[0].pid, *arcs[0].worker_pids]
    assert len(pids) == 2
    assert _gone(pids)


_HANG_SCRIPT = """
import json, os, signal, time
from functools import partial
from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator
from lunarops.classes.dynamics.arc_process import BackwardArcProcess
from tests.dynamics import arc_cases
from tests.parallel import probe_handlers
from tests.support.dynamics import EPOCH


class Dying(arc_cases.RecordingDiagnostic):
    def __call__(self, epoch, *args):
        super().__call__(epoch, *args)
        if EPOCH.seconds_until(epoch) > 0.3 * 86400.0:  # after the seed (node 76 of 360)
            while not arc.worker_pids:
                arc.poll()
                time.sleep(0.01)
            print(json.dumps([arc.pid, *arc.worker_pids]), flush=True)
            os.kill(os.getpid(), signal.SIGKILL)


settings, system, state, spans, options = arc_cases.CASES["advanced-tide"]()
diagnostic = Dying()
diagnostic.bind(system)
factory = partial(arc_cases.setup, "advanced-tide")
with BackwardArcProcess(factory, workers=1, initializer=probe_handlers.register_test_factories) as arc:
    arc.start()
    AdamsBashforthMoultonIntegrator(settings).integrate(
        system, EPOCH, state, spans["duration"], backward_duration_s=spans["back"], node_diagnostic=diagnostic,
        backward_arc=arc, **options,
    )
"""


def test_the_backward_process_and_its_workers_exit_when_the_parent_is_killed():
    result = subprocess.run(
        [sys.executable, "-c", _HANG_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == -signal.SIGKILL, result.stdout + result.stderr
    pids = json.loads(result.stdout.splitlines()[-1])
    assert len(pids) == 2
    assert _gone(pids)


_MPI_RUNNER = """
import json
import numpy as np
from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics.epoch_data import SerialEpochSource
from lunarops.config.registry import register_factory
from lunarops.parallel import mpi
from lunarops.parallel.epoch_stream import RuntimeEpochSource
from tests.support.dynamics import EPOCH, History, build, solar_system

register_factory("ephemerides", "testAnalyticSolarSystem", lambda config, context: solar_system())
runtime = mpi.MpiRuntime()
if runtime.is_master:
    system, state = build(fused=True, rotation=False)
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=600.0, order=4, startup_step_s=75.0))

    def run(source):
        calls = []

        def record(epoch, node_state, batch, slot, history):
            derivative = system.derivatives(epoch, node_state, batch, index=slot, history=history)
            calls.append((float(EPOCH.seconds_until(epoch)), node_state.tolist(), derivative.tolist()))

        trajectory = integrator.integrate(
            system, EPOCH, state, 43200.0, backward_duration_s=172800.0, external_history_provider=History(state),
            epoch_source=source, node_diagnostic=record,
        )
        return trajectory.states, calls

    serial = run(SerialEpochSource(system.epoch_inputs, system.ephemeris))
    workers = run(RuntimeEpochSource(runtime, system.epoch_inputs, {"type": "testAnalyticSolarSystem"}, ".", "arcs"))
    identical = np.array_equal(serial[0], workers[0]) and serial[1] == workers[1]
    runtime.shutdown()
    print(json.dumps({"identical": identical}), flush=True)
else:
    runtime.worker_loop()
"""


@pytest.mark.mpi
def test_a_sequential_two_sided_run_on_mpi_workers_is_bitwise_serial():
    result = subprocess.run(
        [mpi_launcher(), "-n", "3", sys.executable, "-c", _MPI_RUNNER],
        cwd=REPO_ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"identical": True}
