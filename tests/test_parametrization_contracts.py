import json

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import Parametrization, ParametrizationList
from lunarops.classes.parametrization.station_range_bias import StationRangeBiasParametrization
from lunarops.classes.time import Epoch, TimeScale


def _eq(station="APOLLO"):
    return EquationBatch.single(
        oc=0.0,
        sigma=1.0,
        observation_id=station,
        station=station,
        reflector="REF",
        epoch=Epoch.from_isot("2008-01-01T00:00:00", scale=TimeScale.UTC),
        receive_epoch=Epoch.from_isot("2008-01-01T00:00:00", scale=TimeScale.UTC),
    )


def assert_parametrization_contract(block: Parametrization, equations, context=None):
    equations = EquationBatch.concatenate(equations)
    block.setup(equations)
    names1 = block.names
    names2 = block.names
    assert names1 == names2
    design = np.zeros((len(equations), len(names1)))
    block.design(equations, design)
    assert design.shape == (len(equations), len(names1))
    assert block.x0() == pytest.approx(np.zeros(len(names1)))
    json.dumps(block.state(), default=str)


def test_station_range_bias_parametrization_contract():
    assert_parametrization_contract(StationRangeBiasParametrization(per="station"), [_eq("APOLLO")])


class OtherParametrization(Parametrization):
    TYPE = "other"

    def __init__(self):
        self.names = [ParameterName("other", "position.x")]

    def design(self, batch, out):
        out[:, 0] = 1.0

    def reduce(self, batch):
        return np.zeros(len(batch))

    def update(self, delta):
        pass


def test_unselected_parametrization_still_reduces_observations():
    equation = _eq("APOLLO")
    bias = StationRangeBiasParametrization(per="station")
    other = OtherParametrization()
    parametrizations = ParametrizationList([bias, other])
    parametrizations.setup(equation)
    bias.update(np.array([0.25]))

    selected = parametrizations.select_blocks((other.block_id,))
    selected.setup(equation)

    assert selected.names == other.names
    assert selected.reduce(equation) == pytest.approx(-0.25)
