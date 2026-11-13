"""Two-sided propagation with the backward arc in a forked process."""

import json
import multiprocessing
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading

import numpy as np
import pytest

from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings, parallel_arcs
from lunarops.classes.dynamics.epoch_data import SerialEpochSource
from tests.conftest import REPO_ROOT, mpi_launcher
from tests.dynamics.test_deformation import delayed_system
from tests.dynamics.test_translation import CIRCULAR, _two_body
from tests.support.dynamics import EPOCH, History, build, solar_system

DAY = 86400.0

pytestmark = [
    pytest.mark.skipif(not parallel_arcs.parallel_arcs_available(), reason="needs Linux, fork and two CPUs"),
    # A failed fork falls back to the sequential arc with a warning: it must not pass a test unnoticed.
    pytest.mark.filterwarnings("error::RuntimeWarning"),
]

_FORKS: list[int] = []
"""Number of threads of this process at each fork."""

os.register_at_fork(before=lambda: _FORKS.append(threading.active_count()))


@pytest.fixture(autouse=True)
def _no_thread_left_over():
    yield
    assert threading.active_count() == 1, threading.enumerate()


def _two_body_case():
    settings = IntegratorSettings(step_s=1800.0, order=8)
    return settings, _two_body(), CIRCULAR, {"duration": 5 * DAY, "back": 8 * DAY}, {}


def _deformation_case():
    # Lunar delay 0.145 d at 2700 s: the backward arc extrapolates t - tau with the widened stencil.
    settings = IntegratorSettings(step_s=2700.0, order=13, history_interpolation_order=13, startup_step_s=337.5)
    system, state = delayed_system(delay_s=0.14519388 * DAY)
    return settings, system, state, {"duration": 3 * DAY, "back": 4 * DAY}, {"seed_duration_s": DAY}


def _tide_deformation_case():
    # Rotation, deformation and the Earth tide: the history counters of both arcs are non-zero.
    settings = IntegratorSettings(step_s=2700.0, order=13, history_interpolation_order=13, startup_step_s=337.5)
    system, state = build(fused=True, rotation=True, deformation=True)
    options = {"external_history_provider": History(state), "seed_duration_s": DAY}
    return settings, system, state, {"duration": 3 * DAY, "back": 4 * DAY}, options


def _relay_case(step_s=600.0):
    # External epoch source: the backward batches are relayed through this process. At 600 s the advanced
    # semidiurnal tide argument reaches 31 nodes behind the first backward step. The backward arc is the
    # longer one, so part of it is relayed after the forward arc has finished.
    settings = IntegratorSettings(step_s=step_s, order=4, startup_step_s=step_s / 8)
    system, state = build(fused=True, rotation=False)
    source = SerialEpochSource(pickle.loads(pickle.dumps(system.epoch_inputs)), solar_system())
    options = {"external_history_provider": History(state), "epoch_source": source}
    return settings, system, state, {"duration": DAY / 2, "back": 2 * DAY}, options


def _stashed_relay_case():
    # At 240 s the backward arc mirrors 77 forward nodes, so the first backward batch is fetched before the
    # fork and handed to the child with it.
    return _relay_case(step_s=240.0)


CASES = {
    "two-body": _two_body_case,
    "deformation": _deformation_case,
    "tide-deformation": _tide_deformation_case,
    "relay": _relay_case,
    "stashed-relay": _stashed_relay_case,
}


def _propagate(case, *, parallel, diagnostic=True, forks=None):
    settings, system, state, spans, options = CASES[case]()
    calls = []

    def record(epoch, node_state, batch, slot, history):
        # Re-evaluating the dynamics reads the history of the arc, as the acceleration diagnostics do.
        derivative = system.derivatives(epoch, node_state, batch, index=slot, history=history)
        counters = (history.extrapolated_translation_queries, history.max_translation_extrapolation_s)
        calls.append((float(EPOCH.seconds_until(epoch)), node_state.copy(), derivative, slot, counters))

    first = len(_FORKS)
    trajectory = AdamsBashforthMoultonIntegrator(settings).integrate(
        system,
        EPOCH,
        state,
        spans["duration"],
        backward_duration_s=spans["back"],
        node_diagnostic=record if diagnostic else None,
        parallel_arcs=parallel,
        **options,
    )
    # One fork per parallel run, from a process without other threads.
    assert _FORKS[first:] == [1] * (int(parallel) if forks is None else forks)
    return trajectory, calls


def _assert_same_calls(got, expected):
    assert [call[0] for call in got] == [call[0] for call in expected]
    for (_, state, derivative, slot, counters), (_, state0, derivative0, slot0, counters0) in zip(
        got, expected, strict=True
    ):
        np.testing.assert_array_equal(state, state0)
        np.testing.assert_array_equal(derivative, derivative0)
        assert (slot, counters) == (slot0, counters0)


@pytest.mark.parametrize("case", sorted(CASES))
def test_parallel_arcs_are_bitwise_identical_to_the_sequential_propagation(case):
    sequential, sequential_calls = _propagate(case, parallel=False)
    parallel, parallel_calls = _propagate(case, parallel=True)

    np.testing.assert_array_equal(parallel.times_s, sequential.times_s)
    np.testing.assert_array_equal(parallel.states, sequential.states)
    if sequential.history is not None:
        counters = (
            sequential.history.extrapolated_translation_queries,
            sequential.history.max_translation_extrapolation_s,
        )
        assert (
            parallel.history.extrapolated_translation_queries,
            parallel.history.max_translation_extrapolation_s,
        ) == (counters)
    # Every node is reported once, each arc in its integration order, with the same batch slot and the history
    # counters a sequential run shows at that node; the arcs may interleave.
    for arc in (lambda offset: offset >= 0, lambda offset: offset < 0):
        _assert_same_calls(
            [call for call in parallel_calls if arc(call[0])], [call for call in sequential_calls if arc(call[0])]
        )


def test_the_history_counters_are_exercised():
    sequential, _ = _propagate("tide-deformation", parallel=False)
    assert sequential.history._backward.extrapolated_translation_queries > 0
    assert sequential.history._forward.extrapolated_translation_queries > 0


@pytest.mark.parametrize("case", ["two-body", "relay", "stashed-relay"])
def test_a_failed_fork_propagates_the_backward_arc_after_the_forward_one(case, monkeypatch):
    sequential, sequential_calls = _propagate(case, parallel=False)

    def no_fork(process):
        raise OSError("fork refused")

    monkeypatch.setattr(multiprocessing.get_context("fork").Process, "start", no_fork)
    with pytest.warns(RuntimeWarning, match="fork refused"):
        fallback, fallback_calls = _propagate(case, parallel=True, forks=0)
    np.testing.assert_array_equal(fallback.states, sequential.states)
    _assert_same_calls(fallback_calls, sequential_calls)


def test_parallel_arcs_without_a_node_diagnostic():
    sequential, _ = _propagate("relay", parallel=False, diagnostic=False)
    parallel, _ = _propagate("relay", parallel=True, diagnostic=False)
    np.testing.assert_array_equal(parallel.states, sequential.states)


def _fail_on_the_backward_arc(monkeypatch, action, after_s=None):
    """Run ``action`` instead of the ABM step beyond ``after_s`` (five steps by default) into the backward arc.

    The failure must happen inside the child at a chosen node, which only the integrator's step can provide.
    """
    compute = AdamsBashforthMoultonIntegrator._compute_abm_step

    def failing(self, system, nodes, history, epoch, *args):
        if EPOCH.seconds_until(epoch) < -(5 * self.settings.step_s if after_s is None else after_s):
            action()
        return compute(self, system, nodes, history, epoch, *args)

    monkeypatch.setattr(AdamsBashforthMoultonIntegrator, "_compute_abm_step", failing)


def _assert_no_child_left():
    assert not multiprocessing.active_children()


def test_an_error_on_the_backward_arc_is_raised_with_its_remote_traceback(monkeypatch):
    def fail():
        raise ValueError("backward arc failed")

    _fail_on_the_backward_arc(monkeypatch, fail)
    with pytest.raises(ValueError, match="backward arc failed") as raised:
        _propagate("two-body", parallel=True)
    assert "in fail" in str(raised.value.__cause__)
    _assert_no_child_left()


def test_an_unpicklable_error_on_the_backward_arc_is_reported_by_name(monkeypatch):
    class LocalError(Exception):
        pass

    def fail():
        raise LocalError("not importable")

    _fail_on_the_backward_arc(monkeypatch, fail)
    with pytest.raises(RuntimeError, match="LocalError: not importable"):
        _propagate("two-body", parallel=True)
    _assert_no_child_left()


@pytest.mark.parametrize("case", ["two-body", "relay"])
def test_a_backward_process_that_dies_is_reported(case, monkeypatch):
    _fail_on_the_backward_arc(monkeypatch, lambda: os._exit(3))
    with pytest.raises(RuntimeError, match="exited with code 3"):
        _propagate(case, parallel=True)
    _assert_no_child_left()


def test_a_backward_process_that_dies_inside_a_message_is_reported(monkeypatch):
    # The header announces 1 MiB; the parent must see the end of the pipe instead of waiting for the rest.
    def partial(sender, message):
        os.write(sender._connection.fileno(), struct.pack("!i", 1 << 20) + b"partial")
        os._exit(9)

    monkeypatch.setattr(parallel_arcs._Sender, "send", partial)
    with pytest.raises(RuntimeError, match="exited with code 9"):
        _propagate("two-body", parallel=True)
    _assert_no_child_left()


class _Deadlock(Exception):
    pass


def test_a_backward_error_during_the_relay_is_reported_with_full_pipes(monkeypatch):
    # With one-page pipes the child's unread nodes fill its results pipe while this process waits to relay the
    # next batch: the failing child must release that wait instead of both sides waiting for ever.
    monkeypatch.setattr(parallel_arcs, "_PIPE_BYTES", 4096)

    def fail():
        raise ValueError("relay-phase failure")

    _fail_on_the_backward_arc(monkeypatch, fail, after_s=1.5 * DAY)

    def hung(signum, frame):
        raise _Deadlock("the relay deadlocked")  # not an OSError, which the relay handles

    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(30)
    try:
        with pytest.raises(ValueError, match="relay-phase failure"):
            _propagate("relay", parallel=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    _assert_no_child_left()


_MPI_RUNNER = """
import json
import os
import numpy as np
from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics.parallel_arcs import parallel_arcs_available
from lunarops.config.registry import register_factory
from lunarops.parallel import mpi
from lunarops.parallel.epoch_stream import MpiEpochSource
from tests.support.dynamics import EPOCH, History, build, solar_system

register_factory("ephemerides", "testAnalyticSolarSystem", lambda config, context: solar_system())
runtime = mpi.MpiRuntime()
if runtime.is_master:
    assert parallel_arcs_available(), "rank 0 must see two CPUs"
    forks = []
    os.register_at_fork(before=lambda: forks.append(1))
    system, state = build(fused=True, rotation=False)
    source = MpiEpochSource(runtime, system.epoch_inputs, {"type": "testAnalyticSolarSystem"}, ".", "arcs")
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=600.0, order=4, startup_step_s=75.0))

    def run(parallel):
        calls = []

        def record(epoch, node_state, batch, slot, history):
            derivative = system.derivatives(epoch, node_state, batch, index=slot, history=history)
            calls.append((float(EPOCH.seconds_until(epoch)), node_state.tolist(), derivative.tolist()))

        trajectory = integrator.integrate(
            system, EPOCH, state, 43200.0, backward_duration_s=172800.0, external_history_provider=History(state),
            epoch_source=source, node_diagnostic=record, parallel_arcs=parallel,
        )
        return trajectory.states, sorted(calls)

    sequential, parallel = run(False), run(True)
    identical = np.array_equal(sequential[0], parallel[0]) and sequential[1] == parallel[1] and len(forks) == 1
    compute = AdamsBashforthMoultonIntegrator._compute_abm_step

    def dying(self, system, nodes, history, epoch, *args):
        if EPOCH.seconds_until(epoch) < -3000.0:
            os._exit(3)
        return compute(self, system, nodes, history, epoch, *args)

    AdamsBashforthMoultonIntegrator._compute_abm_step = dying
    try:
        run(True)
    except RuntimeError as error:
        reported = "exited with code 3" in str(error)
    else:
        reported = False
    AdamsBashforthMoultonIntegrator._compute_abm_step = compute
    # The stream of the failed run was drained: the workers serve the next one.
    recovered = np.array_equal(run(True)[0], sequential[0])
    runtime.shutdown()
    print(json.dumps({"identical": identical, "reported": reported, "recovered": recovered}), flush=True)
else:
    runtime.worker_loop()
"""


@pytest.mark.mpi
def test_parallel_arcs_relay_the_mpi_epoch_stream_bitwise():
    result = subprocess.run(
        # Unbound, so that rank 0 has a second CPU for the backward-arc process.
        [mpi_launcher(), "--bind-to", "none", "-n", "3", sys.executable, "-c", _MPI_RUNNER],
        cwd=REPO_ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"identical": True, "reported": True, "recovered": True}


def test_an_error_on_the_forward_arc_stops_the_backward_process(monkeypatch):
    settings, system, state, spans, options = CASES["relay"]()

    def fail_late(epoch, *args):
        if EPOCH.seconds_until(epoch) > spans["duration"] / 2:
            raise KeyError("forward diagnostic failed")

    with pytest.raises(KeyError, match="forward diagnostic failed"):
        AdamsBashforthMoultonIntegrator(settings).integrate(
            system,
            EPOCH,
            state,
            spans["duration"],
            backward_duration_s=spans["back"],
            node_diagnostic=fail_late,
            **options,
        )
    _assert_no_child_left()
