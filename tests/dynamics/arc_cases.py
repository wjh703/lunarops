"""Two-sided propagation cases for the backward-arc process tests.

The backward-arc process is spawned: it sees neither the closures nor the monkeypatches of the test process.
It builds each case again from the module-level factories here, and failures are injected inside it by the
factory itself (``failing_setup``).
"""

import os
import time
from dataclasses import replace

from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics.arc_process import ArcSetup
from tests.dynamics.test_deformation import delayed_system
from tests.dynamics.test_translation import CIRCULAR, _two_body
from tests.support.dynamics import EPOCH, History, build

DAY = 86400.0

EPOCH_SOURCE_SPEC = {"ephemerides": {"type": "testAnalyticSolarSystem"}, "workingDir": "."}
"""Ephemeris of ``build`` for worker processes that run ``probe_handlers.register_test_factories``."""


def _two_body_case():
    settings = IntegratorSettings(step_s=1800.0, order=8)
    return settings, _two_body(), CIRCULAR, {"duration": 5 * DAY, "back": 8 * DAY}, {}


def _single_mirror_case():
    # Order 1 without advanced delays: the backward arc mirrors only the initial node, so it is seeded before
    # the forward main grid starts.
    settings = IntegratorSettings(step_s=600.0, order=1, startup_step_s=75.0)
    return settings, _two_body(), CIRCULAR, {"duration": DAY / 4, "back": DAY / 2}, {}


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


def _advanced_tide_case():
    # At 240 s the advanced semidiurnal tide argument makes the backward arc mirror 77 forward nodes: it is
    # seeded in the second forward epoch batch. The backward arc is the longer one.
    settings = IntegratorSettings(step_s=240.0, order=4, startup_step_s=30.0)
    system, state = build(fused=True, rotation=False)
    return (
        settings,
        system,
        state,
        {"duration": DAY / 2, "back": 2 * DAY},
        {"external_history_provider": History(state)},
    )


CASES = {
    "two-body": _two_body_case,
    "single-mirror": _single_mirror_case,
    "deformation": _deformation_case,
    "tide-deformation": _tide_deformation_case,
    "advanced-tide": _advanced_tide_case,
}
"""Cases by name: each returns ``(settings, system, initial_state, spans, integrate_options)``."""


def setup(case: str) -> ArcSetup:
    """Build a case in the backward-arc process."""
    settings, system, *_ = CASES[case]()
    return ArcSetup(system, AdamsBashforthMoultonIntegrator(settings), epoch_source_spec=EPOCH_SOURCE_SPEC)


def _fail():
    raise ValueError("backward arc failed")


def _fail_unpicklably():
    class LocalError(Exception):
        pass

    raise LocalError("not importable")


ACTIONS = {
    "raise": _fail,
    "unpicklable": _fail_unpicklably,
    "exit": lambda: os._exit(3),
    "stall": lambda: time.sleep(3600.0),
}


def failing_setup(case: str, action: str, after_steps: int = 5) -> ArcSetup:
    """Build a case whose ABM steps run ``ACTIONS[action]`` beyond ``after_steps`` steps into the backward arc.

    Only the integrator's step fails at a chosen node; the patch lives in the backward-arc process only.
    """
    compute = AdamsBashforthMoultonIntegrator._compute_abm_step

    def failing(self, system, nodes, history, epoch, *args):
        if EPOCH.seconds_until(epoch) < -after_steps * self.settings.step_s:
            ACTIONS[action]()
        return compute(self, system, nodes, history, epoch, *args)

    AdamsBashforthMoultonIntegrator._compute_abm_step = failing
    return setup(case)


def broken_setup(case: str) -> ArcSetup:
    """Fail to build a case."""
    raise ValueError(f"cannot build {case}")


def mismatched_setup(case: str) -> ArcSetup:
    """Build a case with an integrator that differs from the parent's."""
    settings, system, *_ = CASES[case]()
    changed = replace(settings, corrector_iterations=settings.corrector_iterations + 1)
    return ArcSetup(system, AdamsBashforthMoultonIntegrator(changed))


class RecordingDiagnostic:
    """Node diagnostic that records every call, re-evaluating the dynamics as the acceleration diagnostics do."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._system = None

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_system": None}

    def bind(self, system) -> None:
        self._system = system

    def for_backward_arc(self) -> RecordingDiagnostic:
        return RecordingDiagnostic()

    def merge_backward(self, other: RecordingDiagnostic) -> None:
        self.calls.extend(other.calls)

    def __call__(self, epoch, node_state, batch, slot, history) -> None:
        # Re-evaluating the dynamics reads the history of the arc, which may move its counters.
        derivative = self._system.derivatives(epoch, node_state, batch, index=slot, history=history)
        counters = (history.extrapolated_translation_queries, history.max_translation_extrapolation_s)
        self.calls.append((float(EPOCH.seconds_until(epoch)), node_state.copy(), derivative, slot, counters))
