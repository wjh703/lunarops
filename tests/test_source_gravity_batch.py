"""Batched source/source Newtonian kernel parity with the per-epoch kernel."""

import numpy as np
import pytest

from lunarops._dynamics_core import point_mass_symmetric, point_mass_symmetric_batch


@pytest.mark.parametrize("epochs,bodies", [(0, 4), (1, 0), (3, 1), (64, 128)])
def test_batch_preserves_scalar_pair_order(epochs, bodies):
    rng = np.random.default_rng(617)
    positions = rng.normal(size=(epochs, bodies, 3)) * 1e11
    gm = np.exp(rng.uniform(10, 45, size=(epochs, bodies)))
    acceleration, potential = point_mass_symmetric_batch(positions, gm)
    assert acceleration.shape == positions.shape
    assert potential.shape == gm.shape
    for index in range(epochs):
        expected_acceleration, expected_potential = point_mass_symmetric(positions[index], gm[index])
        np.testing.assert_array_equal(acceleration[index], expected_acceleration)
        np.testing.assert_array_equal(potential[index], expected_potential)


@pytest.mark.parametrize("bad_epoch", [0, 3, 7])
def test_coincident_pair_is_reported_from_any_parallel_epoch(bad_epoch):
    positions = np.random.default_rng(118).normal(size=(8, 64, 3))
    positions[bad_epoch, 1] = positions[bad_epoch, 0]
    with pytest.raises(ValueError, match="coincident"):
        point_mass_symmetric_batch(positions, np.ones((8, 64)))
