"""The integrator rejects backward-arc nodes that are not those of the backward grid."""

import numpy as np
import pytest

from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics.arc_process import ArcResult
from lunarops.classes.dynamics.force_models import NewtonianPointMassForce
from tests.support.dynamics import EPOCH, circular_moon_ephemeris, make_system

SETTINGS = IntegratorSettings(step_s=60.0, order=3, startup_step_s=15.0)
STATE = np.array([3.844e8, 0.0, 0.0, 0.0, 1000.0, 0.0])
BACKWARD_TIMES = -60.0 * np.arange(1, 6)


class _ReturnedArc:
    """Backward arc that returns fixed nodes instead of propagating them."""

    def __init__(self, times, dimension=6):
        self._nodes = (times, *(np.zeros((dimension, len(times))) for _ in range(4)))

    def seed(self, *args):
        pass

    def poll(self, progress=None):
        pass

    def finish(self, progress=None):
        return ArcResult(self._nodes, 0, 0.0, None)


@pytest.mark.parametrize(
    ("arc", "message"),
    [
        (_ReturnedArc(BACKWARD_TIMES[:-1]), "returned 4 nodes; expected 5"),
        (_ReturnedArc(BACKWARD_TIMES + 1e-9), "at other times than the backward grid"),
        (_ReturnedArc(BACKWARD_TIMES[::-1]), "at other times than the backward grid"),
        (_ReturnedArc(BACKWARD_TIMES, dimension=12), r"shape \(12, 5\); expected \(6, 5\)"),
    ],
)
def test_backward_nodes_off_the_grid_are_rejected(arc, message):
    system = make_system(rotation=None, forces=(NewtonianPointMassForce(),), ephemeris=circular_moon_ephemeris())
    integrator = AdamsBashforthMoultonIntegrator(SETTINGS)
    with pytest.raises(RuntimeError, match=message):
        integrator.integrate(system, EPOCH, STATE, 600.0, backward_duration_s=300.0, backward_arc=arc)
