"""Parameter partials of the variational equations against differences of whole propagations."""

import numpy as np
import pytest

from lunarops.classes.dynamics.assembly import build_lunar_propagation_problem
from lunarops.classes.dynamics.parameters import (
    DYNAMICAL_PARAMETER_NAMES,
    LOCAL_INCREMENTS,
    build_parameter_variations,
)
from lunarops.classes.dynamics.variational import VariationalEquations
from tests.dynamics.test_parameters import _full_config

# Scales that make the state dimensionless: position, velocity, Euler angle, angular velocity.
SCALES = np.r_[[3.84e8] * 3, [1.0e3] * 3, [1.0] * 3, [2.7e-6] * 6]
# Half-widths of the differences of whole propagations, in units of the local increments.
GLOBAL_SCALE = {"tau_r1": 30.0, "tau_r2": 30.0, "tau_m": 3.0}


def _run(problem, variational=None, backward=0.0):
    return problem.integrator.integrate(
        problem.dynamics,
        problem.initial_epoch,
        problem.initial_vector,
        problem.duration_s,
        backward_duration_s=backward,
        seed_duration_s=problem.seed_duration_s,
        external_history_provider=problem.external_history,
        variational=variational,
    )


def _nodes(problem, trajectory, times):
    return np.asarray([trajectory.vector(problem.initial_epoch.shifted(float(offset))) for offset in times]).T


@pytest.mark.parametrize("backward", [0.0, 3600.0])
def test_parameter_partials_match_differences_of_propagations(tmp_path, backward):
    config, context = _full_config(tmp_path, durationSeconds=7200, correctorIterations=1)
    problem = build_lunar_propagation_problem(config, context)
    variations = build_parameter_variations(problem, context)
    assert tuple(variation.name for variation in variations) == DYNAMICAL_PARAMETER_NAMES
    variational = VariationalEquations(
        problem.dynamics, problem.integrator.settings, parameters=variations, output_stride=10
    )
    plain = _run(problem, backward=backward)
    trajectory = _run(problem, variational, backward=backward)
    times, partials = variational.times, variational.partials
    dimension = variational.dimension
    assert partials.shape == (len(times), dimension, dimension + len(variations))
    assert times[0] == -backward and times[-1] == problem.duration_s
    np.testing.assert_array_equal(_nodes(problem, trajectory, times), _nodes(problem, plain, times))

    for column, variation in enumerate(variations, start=dimension):
        name = variation.name
        increment = LOCAL_INCREMENTS[name] * GLOBAL_SCALE.get(name, 10.0)
        states = [
            _nodes(
                problem,
                _run(
                    build_lunar_propagation_problem(
                        config,
                        context,
                        parameters=problem.parameters.with_value(name, problem.parameters[name] + sign * increment),
                    ),
                    backward=backward,
                ),
                times,
            )
            for sign in (1.0, -1.0)
        ]
        difference = (states[0] - states[1]) / (2 * increment) / SCALES[:, None]
        error = np.abs(partials[:, :, column].T / SCALES[:, None] - difference).max()
        reference = np.abs(difference).max()
        print(f"{name:7s} max {reference:.3e} error {error / reference:.2e}")
        assert reference > 0
        # The differences carry the rounding of the two states, about 1e-16 of each scale.
        assert error < 1.0e-6 * reference + 2.0e-16 / increment, name


def test_parameter_columns_need_their_variations_to_rebind(tmp_path):
    config, context = _full_config(tmp_path)
    problem = build_lunar_propagation_problem(config, context)
    variations = build_parameter_variations(problem, context, ["fc", "kv_ct"])
    variational = VariationalEquations(problem.dynamics, problem.integrator.settings, parameters=variations)
    assert variational.parameter_names == ("fc", "kv_ct")
    assert variational.columns == variational.dimension + 2
    mirrored = object.__new__(VariationalEquations)
    mirrored.__dict__.update(variational.__getstate__())
    with pytest.raises(ValueError, match="parameter variations"):
        mirrored.bind(problem.dynamics)
    with pytest.raises(ValueError, match="differ"):
        mirrored.bind(problem.dynamics, variations[::-1])
    mirrored.bind(problem.dynamics, variations)
    with pytest.raises(ValueError, match="not a dynamical parameter"):
        build_parameter_variations(problem, context, ["r0_x"])
    with pytest.raises(ValueError, match="distinct"):
        VariationalEquations(problem.dynamics, problem.integrator.settings, parameters=[variations[0], variations[0]])
