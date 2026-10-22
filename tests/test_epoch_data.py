"""Epoch inputs built by MPI workers from a pickled recipe equal the in-process ones bit for bit."""

import pickle

import numpy as np
from test_fused_rhs import EPOCH, History, SolarSystem, build

from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics.epoch_data import SerialEpochSource, batch_epochs


def test_pickled_epoch_inputs_reproduce_arrays():
    system, _ = build(fused=True, rotation=False)
    recipe = pickle.loads(pickle.dumps(system.epoch_inputs))
    epochs = batch_epochs(EPOCH, [0.0, 2700.0, -5400.0, 1.0e7])
    local = system.epoch_inputs.arrays(system.ephemeris, epochs)
    remote = recipe.arrays(SolarSystem(), epochs)
    assert set(local) == set(remote)
    for key, value in local.items():
        if value is None:
            assert remote[key] is None
        else:
            np.testing.assert_array_equal(remote[key], value)
    assert local["delayedStates"].shape == (4, 3, 2, 6)


def test_integrator_accepts_an_external_epoch_source():
    class Counting(SerialEpochSource):
        batches = 0

        def stream(self, initial_epoch, offset_batches):
            for arrays in super().stream(initial_epoch, offset_batches):
                Counting.batches += 1
                yield arrays

    system, state = build(fused=True, rotation=False)
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=600.0, order=4, startup_step_s=75.0))
    history = History(state)
    default = integrator.integrate(system, EPOCH, state, 86400.0, external_history_provider=history)
    source = Counting(pickle.loads(pickle.dumps(system.epoch_inputs)), SolarSystem())
    streamed = integrator.integrate(
        system, EPOCH, state, 86400.0, external_history_provider=history, epoch_source=source
    )
    assert Counting.batches == 3
    np.testing.assert_array_equal(streamed.states, default.states)
