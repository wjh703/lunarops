import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName, parameter_unit
from lunarops.estimation.normal_equation_solver import (
    NormalEquationSingularError,
    solve_normal_equations,
)
from lunarops.estimation.normal_equations import NormalEquations


def test_singular_normal_equations_raise_diagnostic_error():
    names = [ParameterName("x", "position.x"), ParameterName("y", "position.y")]
    normals = NormalEquations(names, np.zeros((2, 2)), np.zeros(2), meta={},
                              parameter_units=[parameter_unit(name) for name in names], x0=np.zeros(2))
    normals.N[:] = np.array([[1.0, 1.0], [1.0, 1.0]])
    normals.W[:] = np.array([1.0, 1.0])
    normals.obs_count = 2

    with pytest.raises(NormalEquationSingularError) as err:
        solve_normal_equations(normals)

    message = str(err.value)
    assert "rank=1/2" in message
    assert "obs_count=2" in message


def test_normal_matrix_diagnostics_are_scale_aware_and_report_true_condition_number():
    names = [ParameterName("x", "position.x"), ParameterName("y", "position.y")]
    small = NormalEquations(names, np.zeros((2, 2)), np.zeros(2),
                              meta={}, parameter_units=[parameter_unit(name) for name in names], x0=np.zeros(len(names)))
    small.N[:] = np.diag([1.0e-30, 2.0e-30])
    small.W[:] = np.array([3.0e-30, 8.0e-30])
    small.lPl = 4.1e-29
    small.obs_count = 3

    assert solve_normal_equations(small).values == pytest.approx([3.0, 4.0])


def test_positive_definite_but_numerically_unobservable_system_is_rejected():
    names = [ParameterName("x", "position.x"), ParameterName("y", "position.y")]
    normals = NormalEquations(names, np.diag([1.0, 1.0e-20]), np.zeros(2), obs_count=3,
                              meta={}, parameter_units=[parameter_unit(name) for name in names], x0=np.zeros(len(names)))
    with pytest.raises(NormalEquationSingularError, match="rank=1/2"):
        solve_normal_equations(normals)
