import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.normal_equation_solver import (
    NormalEquationSingularError,
    normal_matrix_condition,
    normal_matrix_rank,
    solve_normal_equations,
)
from lunarops.estimation.normal_equations import NormalEquations


def test_singular_normal_equations_raise_diagnostic_error():
    normals = NormalEquations.zeros([ParameterName("x", "position.x"), ParameterName("y", "position.y")])
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
    small = NormalEquations.zeros(names)
    small.N[:] = np.diag([1.0e-30, 2.0e-30])
    small.W[:] = np.array([3.0e-30, 8.0e-30])
    small.lPl = 4.1e-29
    small.obs_count = 3

    assert normal_matrix_rank(small) == 2
    assert normal_matrix_condition(small) == pytest.approx(2.0)
    assert solve_normal_equations(small).delta == pytest.approx([3.0, 4.0])

    singular = NormalEquations.zeros(names)
    singular.N[:] = np.diag([1.0, 0.0])
    singular.obs_count = 2

    assert normal_matrix_rank(singular) == 1
    assert normal_matrix_condition(singular) == float("inf")

    zero = NormalEquations.zeros(names)
    assert normal_matrix_rank(zero) == 0
    assert normal_matrix_condition(zero) == float("inf")
