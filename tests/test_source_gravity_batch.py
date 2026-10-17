"""Batch/source cache parity, including epoch-dependent GM and EIH terms."""

import numpy as np
import pytest

from lunarops._dynamics_core import point_mass_symmetric, point_mass_symmetric_batch
from lunarops.classes.dynamics.forces import (
    build_point_mass_gravity_cache,
    build_point_mass_gravity_cache_batch,
    evaluate_eih_correction,
    evaluate_newtonian_point_mass_system,
)


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


def test_batch_cache_preserves_full_newtonian_and_eih_auxiliaries():
    rng = np.random.default_rng(935)
    positions = rng.normal(size=(16, 128, 3)) * 1e11
    velocities = rng.normal(size=positions.shape) * 1e4
    gm = np.exp(rng.uniform(10, 45, size=positions.shape[:2]))
    caches = build_point_mass_gravity_cache_batch(positions, gm, integrated_body_count=2)
    for r, v, mu, cache in zip(positions, velocities, gm, caches, strict=True):
        reference = build_point_mass_gravity_cache(r, mu, integrated_body_count=2)
        np.testing.assert_array_equal(
            cache.prescribed_body_accelerations_mps2, reference.prescribed_body_accelerations_mps2
        )
        np.testing.assert_array_equal(cache.prescribed_body_potentials_m2_s2, reference.prescribed_body_potentials_m2_s2)
        actual = evaluate_newtonian_point_mass_system(r, mu, cache)
        expected = evaluate_newtonian_point_mass_system(r, mu, reference)
        np.testing.assert_array_equal(actual.accelerations_mps2, expected.accelerations_mps2)
        np.testing.assert_array_equal(actual.potentials_m2_s2, expected.potentials_m2_s2)
        np.testing.assert_array_equal(
            evaluate_eih_correction(r, v, mu, actual, np.array([1])),
            evaluate_eih_correction(r, v, mu, expected, np.array([1])),
        )


