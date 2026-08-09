import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.linearized_least_squares import (
    NormalEquationSingularError,
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
