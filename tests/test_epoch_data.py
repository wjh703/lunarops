"""Epoch inputs built by MPI workers from a pickled recipe equal the in-process ones bit for bit."""

import pickle
from dataclasses import fields

import numpy as np
from test_fused_rhs import EPOCH, History, SolarSystem, build

from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics.epoch_data import SerialEpochSource, batch_epochs


def test_pickled_epoch_inputs_reproduce_arrays():
    system, _ = build(fused=True, rotation=False)
    recipe = pickle.loads(pickle.dumps(system.epoch_inputs))
    epochs = batch_epochs(EPOCH, [0.0, 2700.0, -5400.0, 1.0e7])
    local = system.epoch_inputs.build(system.ephemeris, epochs)
    remote = recipe.build(SolarSystem(), epochs)
    for field in fields(local):
        value = getattr(local, field.name)
        assert isinstance(value, np.ndarray)
        np.testing.assert_array_equal(getattr(remote, field.name), value)


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


def test_startup_history_reconstructs_moon_and_preserves_query_order():
    from lunarops.classes.dynamics.history import EphemerisHistory

    class Ephemeris:
        def body_state_matrix_bcrs(self, names, epoch):
            rows = {"EARTH": np.array([1.0, 2.0, -3.0, 4.0, 5.0, 6.0]),
                    "MOON": np.array([0.1, 0.2, -0.3, 0.4, 0.5, 0.6]),
                    "SUN": np.arange(6, dtype=float)}
            return np.asarray([rows[name] for name in names])

    ephemeris = Ephemeris()
    history = EphemerisHistory(ephemeris)
    expected = ephemeris.body_state_matrix_bcrs(("MOON", "SUN", "EARTH"), EPOCH)
    actual = history.states(("MOON", "SUN", "EARTH"), EPOCH)
    np.testing.assert_array_equal(actual[1:], expected[1:])
    earth = expected[2]
    np.testing.assert_array_equal(actual[0], earth + (expected[0] - earth))
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=np.max(np.spacing(abs(earth))))
