from __future__ import annotations

import numpy as np
import pytest

from lunarops.classes.dynamics import LunarDynamicState, UnsupportedLunarDynamics
from lunarops.classes.time import Epoch, TimeScale


def test_lunar_dynamic_state_requires_tdb_and_fixed_shapes():
    epoch = Epoch(2451545.0, 0.0, TimeScale.TDB)
    state = LunarDynamicState(epoch, np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    assert state.attitude_pa2lcrs.shape == (3, 3)
    with pytest.raises(ValueError):
        LunarDynamicState(Epoch(2451545.0, 0.0, TimeScale.UTC), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))


def test_placeholder_dynamics_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        UnsupportedLunarDynamics().state(Epoch(2451545.0, 0.0, TimeScale.TDB))
